#ifndef LLVM_LIB_TRANSFORMS_REGION_MELDER_H
#define LLVM_LIB_TRANSFORMS_REGION_MELDER_H

#include "RegionAnalyzer.h"
#include "SmithWaterman.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/TableGen/Record.h"
#include "llvm/Transforms/IPO/FunctionMerging.h"

namespace llvm {

class InstructionMatch {
public:
  static bool match(Value *V1, Value *V2) {
    if (isa<Instruction>(V1) && isa<Instruction>(V2)) {
      Instruction *I1 = dyn_cast<Instruction>(V1);
      Instruction *I2 = dyn_cast<Instruction>(V2);

      if (I1->getOpcode() == I2->getOpcode() &&
          I1->getOpcode() == Instruction::Br)
        return true;
    }
    return FunctionMerger::match(V1, V2);
  };
};

// scoring function for instruction alignment based on code size reduction
// if instrutions match always return a non-zero value
struct CodeSizeCostModel : public ScoringFunction<Value *> {
  TargetTransformInfo *TTI;
  CodeSizeCostModel(TargetTransformInfo &TTI) : TTI(&TTI) {}

public:
  int operator()(Value *V1, Value *V2) override {
    if (!InstructionMatch::match(V1, V2))
      return 0;

    int SavedSize = 0;
    if (isa<Instruction>(V1)) {
      SavedSize = 1 + TTI->getInstructionCost(
                             dyn_cast<Instruction>(V1),
                             TargetTransformInfo::TargetCostKind::TCK_CodeSize)
                          .getValue()
                          .getValue();
    } else if (isa<BasicBlock>(V1)) {
      SavedSize = 3;
    }
    return SavedSize;
  }

  int gap(int K) override {
    int BrCost =
        TTI->getCFInstrCost(Instruction::Br, TTI::TargetCostKind::TCK_CodeSize)
            .getValue()
            .getValue();
    return 1 + BrCost;
  }
};

// simple scoring function for instruction alignment for latency reduction
struct GPULatencyCostModel : public ScoringFunction<Value *> {
public:
  int operator()(Value *V1, Value *V2) override {

    if (!InstructionMatch::match(V1, V2))
      return 0;

    int SavedCycles = 0;
    if (isa<Instruction>(V1)) {
      Instruction *I1 = dyn_cast<Instruction>(V1);
      SavedCycles = Utils::getInstructionCost(I1);
    } else if (isa<BasicBlock>(V1)) {
      SavedCycles = 3;
    }

    return SavedCycles;
  }
  int gap(int K) override { return 2; }
};

struct RegionMeldingInfo {
public:
  BasicBlock *LEntry{nullptr}, *REntry{nullptr};
  BasicBlock *LExit{nullptr}, *RExit{nullptr};
  bool RegionsSimplified = false;
  DenseMap<BasicBlock *, BasicBlock *> BlockMap;
  BasicBlock *UnifyPHIBlock{nullptr};
  bool ExpandingLeft = false;
  // Expanded Block
  BasicBlock *ExpandedBlock = nullptr;
  // predecessors of melded regions
  SmallVector<BasicBlock *> LPreds, RPreds;
  // successors of melded regions
  SmallVector<BasicBlock *> LSuccs, RSuccs;
  void updateBlockMap(BasicBlock *New, BasicBlock *Old, bool IsLeft);
  void print();
};

class MeldingHandler {
private:
  Function *Func{nullptr};
  Value *DivCond{nullptr};
  TargetTransformInfo &TTI;
  RegionMeldingInfo &MeldingInfo;

  AlignedSeq<Value *> RegionInstrAlignement;
  using ValueToValueMapTy = ValueMap<const Value *, WeakTrackingVH>;

  // value maps
  ValueToValueMapTy MergedValuesToLeftValues;
  ValueToValueMapTy MergedValuesToRightValues;
  ValueToValueMapTy OrigToMergedValues;

  // label map
  DenseMap<BasicBlock *, BasicBlock *> LeftBbToMergedBb;
  DenseMap<BasicBlock *, BasicBlock *> RightBbToMergedBb;

  // merged instructions
  SmallVector<Instruction *, 0> MergedInstructions;
  SmallVector<BasicBlock *, 0> MergedBBs;

  // un-predication
  struct InstrRange {
  private:
    Instruction *StartI, *EndI;
    bool Direction;

  public:
    InstrRange(Instruction *Start, Instruction *End, bool Dir)
        : StartI(Start), EndI(End), Direction(Dir) {}
    Instruction *getStart() const { return StartI; }
    Instruction *getEnd() const { return EndI; }
    bool splitToTrue() const { return Direction; }
    bool splitToFalse() const { return !Direction; }
    friend raw_ostream &operator<<(raw_ostream &OS, const InstrRange &Range) {
      OS << "start : ";
      Range.getStart()->print(OS);
      OS << ", end : ";
      Range.getEnd()->print(OS);
      OS << ", direction : " << (Range.splitToTrue() ? "T" : "F");
      return OS;
    }

    bool canExtendUsing(const InstrRange &Other) {
      return Other.getStart()->getPrevNode() == EndI &&
             Other.splitToTrue() == splitToTrue();
    }
    InstrRange extendUsing(const InstrRange &Other) {
      assert(canExtendUsing(Other) &&
             "this range can not be extended with provided range !");
      return InstrRange(StartI, Other.getEnd(), splitToTrue());
    }
  };

  SmallVector<InstrRange, 16> SplitRanges;

  AlignedSeq<Value *>
  getAlignmentOfBlocks(BasicBlock *LeftBb, BasicBlock *RightBb,
                       ScoringFunction<Value *> &ScoringFunc);
  void computeRegionSeqAlignment(DenseMap<BasicBlock *, BasicBlock *> BbMap);
  void linearizeBb(BasicBlock *BB, SmallVectorImpl<Value *> &LinearizedVals);
  void cloneInstructions();
  void fixOperends();
  void runPostMergeCleanup();
  void setOperends(Instruction *LeftI, Instruction *RightI,
                   Instruction *MergedI);
  void setOperendsForBr(BranchInst *LeftBr, BranchInst *RightBr,
                        BranchInst *MergedBr);
  Instruction *cloneInstruction(Instruction *OrigI, IRBuilder<> &Builder);
  void fixPhiNode(PHINode *Orig);
  void runPostOptimizations();
  void setOprendsForNonMatchingStore(StoreInst *SI, bool IsLeft);

  void runUnpredicationPass();
  void updateSplitRangeMap(bool Direction, Instruction *I);

public:
  MeldingHandler(Function *Func, Value *DivCond, TargetTransformInfo &TTI,
                 RegionMeldingInfo &MeldInfo)
      : Func(Func), DivCond(DivCond), TTI(TTI), MeldingInfo(MeldInfo) {}
  /// merge two SESE regions (or SESE basic blocks)
  void meld();
};

class RegionMelder {
private:
  ControlFlowGraphInfo &CFGInfo;
  RegionAnalysisResult &RAResult;
  RegionMeldingInfo MeldInfo;

  // does region replication. matched block is inside the 'RToReplicate'
  // we replicate RToReplicate and place ExpandedBlock in a corresponding 
  // position to MatchedBlock
  void runRegionReplication(BasicBlock *ExpandedBlock, BasicBlock *MatchedBlock,
                            bool ExpandingLeft, Region *RToReplicate);
  BasicBlock *addUnifyPHIBlock();
  void simplifyRegions();

  // makes the region SESE, after simplification exit block of the region
  // is connected to rest of the CFG with only one edge
  BasicBlock *simplifyRegion(BasicBlock *Exit, BasicBlock *Entry);

  void updateMapping(BasicBlock *NewBb, BasicBlock *OldBb, bool IsLeft);

  bool isInsideMeldedRegion(BasicBlock *BB, BasicBlock *Entry,
                            BasicBlock *Exit);

  // finds the region whose entry block post dominates the path entry block
  Region *getRegionToReplicate(BasicBlock *MatchedBlock, BasicBlock *PathEntry);

public:
  RegionMelder(ControlFlowGraphInfo &CFGInfo, RegionAnalysisResult &RAResult)
      : CFGInfo(CFGInfo), RAResult(RAResult) {}
  /// perform melding on most profitable region pair
  /// returns true if size reduction was positive
  bool run();
};

} // namespace llvm

#endif
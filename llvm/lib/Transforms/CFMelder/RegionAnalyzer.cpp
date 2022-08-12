#include "RegionAnalyzer.h"
#include "SmithWaterman.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <memory>

using namespace llvm;

static cl::opt<double> SimilarityThreshold(
    "cf-merging-similarity-threshold", cl::init(0.2), cl::Hidden,
    cl::desc(
        "Minimum similarity required to merge two basicblocks or two regions"));

static cl::opt<bool> PerformGreedyRegionMatch(
    "use-greedy-region-match", cl::init(false), cl::Hidden,
    cl::desc("Enable greedy region match in control-flow melding"));

static cl::opt<bool> DisableRegionReplication(
    "disable-region-replication", cl::init(false), cl::Hidden,
    cl::desc("Disable replicating regions in control-flow melding"));

static cl::opt<bool>
    RunBranchFusionOnly("run-branch-fusion-only", cl::init(false), cl::Hidden,
                        cl::desc("Run branch fusion only, no region-melding"));

// helper functions

// finds the most similar block in 'Candidates' to block BB
static BasicBlock *
findMostSimilarBb(BasicBlock *BB, SmallVectorImpl<BasicBlock *> &Candidates) {
  assert(Candidates.size() > 0 && "empty basicblock candidate list!");
  BasicBlock *MostSimilar = Candidates[0];
  double BestScore = Utils::computeBlockSimilarity(BB, MostSimilar);
  for (BasicBlock *Candidate : Candidates) {
    double Score = Utils::computeBlockSimilarity(BB, Candidate);
    if (Score > BestScore) {
      MostSimilar = Candidate;
      BestScore = Score;
    }
  }

  return MostSimilar;
}

static void printRegionList(SmallVectorImpl<Region *> &Regions,
                            DominatorTree &DT) {
  for (auto It = Regions.begin(); It != Regions.end(); It++) {

    errs() << "[ ";
    (*It)->getEntry()->printAsOperand(errs(), false);
    errs() << " : ";
    (*It)->getExit()->printAsOperand(errs(), false);
    errs() << " ]";
  }
}

// check if region list is in dominance order
static bool verifyRegionList(SmallVectorImpl<Region *> &Regions,
                             DominatorTree &DT) {
  for (auto It = Regions.begin(); It != Regions.end(); It++) {
    // check if the list is in dominance order
    if ((It + 1) != Regions.end()) {
      Region *Curr = *It;
      Region *Next = *(It + 1);
      if (!DT.dominates(Curr->getEntry(), Next->getEntry()))
        return false;
    }
  }
  return true;
}

void ControlFlowGraphInfo::recompute() {
  DT.recalculate(getFunction());
  PDT.recalculate(getFunction());

  LI = std::make_shared<LoopInfo>(DT);

  /// calculate regions
  DominanceFrontier DF;
  DF.analyze(DT);
  RI = std::make_shared<RegionInfo>();
  RI->recalculate(getFunction(), &DT, &PDT, &DF);
}

ControlFlowGraphInfo::ControlFlowGraphInfo(Function &F, DominatorTree &DT,
                                           PostDominatorTree &PDT,
                                           TargetTransformInfo &TTI)
    : F(F), DT(DT), PDT(PDT), TTI(TTI) {
  LI = std::make_shared<LoopInfo>(DT);
  /// calculate regions
  DominanceFrontier DF;
  DF.analyze(DT);
  RI = std::make_shared<RegionInfo>();
  RI->recalculate(F, &DT, &PDT, &DF);
}

bool RegionComparator::compare() {
  DenseMap<BasicBlock *, int> LabelMapR1;
  DenseMap<BasicBlock *, int> LabelMapR2;
  SmallVector<BasicBlock *, 0> StackR1;
  SmallVector<BasicBlock *, 0> StackR2;

  StackR1.push_back(R1->getEntry());
  StackR2.push_back(R2->getEntry());
  // do a parallel DFS traversal
  while (!StackR1.empty()) {
    BasicBlock *CurrBlockR1 = StackR1.pop_back_val();
    BasicBlock *CurrBlockR2 = StackR2.pop_back_val();
    // num of successors must match
    if (CurrBlockR1->getTerminator()->getNumSuccessors() !=
        CurrBlockR2->getTerminator()->getNumSuccessors())
      return false;
    // add to map
    LabelMapR1.insert(
        std::pair<BasicBlock *, int>(CurrBlockR1, LabelMapR1.size()));
    LabelMapR2.insert(
        std::pair<BasicBlock *, int>(CurrBlockR2, LabelMapR2.size()));
    // update the mapping
    Mapping.insert(
        std::pair<BasicBlock *, BasicBlock *>(CurrBlockR1, CurrBlockR2));

    // iterate over all successors
    for (unsigned I = 0; I < CurrBlockR1->getTerminator()->getNumSuccessors();
         I++) {
      BasicBlock *SuccR1 = CurrBlockR1->getTerminator()->getSuccessor(I);
      BasicBlock *SuccR2 = CurrBlockR2->getTerminator()->getSuccessor(I);
      // check if we already visited this successor
      auto ItR1 = LabelMapR1.find(SuccR1);
      auto ItR2 = LabelMapR2.find(SuccR2);

      int SuccLabelR1 = -1, SuccLabelR2 = -1;

      if (ItR1 != LabelMapR1.end())
        SuccLabelR1 = ItR1->second;

      if (ItR2 != LabelMapR2.end())
        SuccLabelR2 = ItR2->second;

      if (SuccLabelR1 != SuccLabelR2)
        return false;

      // check if exit block is consisitent
      if ((SuccR1 == R1->getExit() && SuccR2 != R2->getExit()) ||
          (SuccR1 != R1->getExit() && SuccR2 == R2->getExit()))
        return false;

      // add succ to stack only if successor is not the exit block and
      // it is not visited before
      if (SuccR1 != R1->getExit() && SuccLabelR1 == -1 &&
          SuccR2 != R2->getExit() && SuccLabelR2 == -1) {
        StackR1.push_back(SuccR1);
        StackR2.push_back(SuccR2);
      }
    }
  }
  return true;
}

void RegionComparator::getMapping(
    DenseMap<BasicBlock *, BasicBlock *> &Result) {
  Result.clear();
  Result.insert(Mapping.begin(), Mapping.end());
}

MergeableRegionPair::MergeableRegionPair(Region &R1, Region &R2,
                                         RegionComparator &Comparator) {
  LEntry = R1.getEntry();
  LExit = R1.getExit();
  REntry = R2.getEntry();
  RExit = R2.getExit();
  Comparator.getMapping(this->Mapping);
  // CalculateSimilarityScore();
  SimilarityScore = Utils::computeRegionSimilarity(Mapping, LExit);
  // errs() << "Similarity score : " << similarityScore << "\n";
}

BasicBlock *MergeableRegionPair::getMatchingRightBb(BasicBlock *BB) {
  auto It = Mapping.find(BB);
  assert(It != Mapping.end() && "Matching Right BB is not found for BB");
  return It->second;
}

bool MergeableRegionPair::dominates(std::shared_ptr<MergeableRegionPair> &Other,
                                    DominatorTree &DT) {
  if (!DT.dominates(getLeftEntry(), Other->getLeftEntry()))
    return false;

  if (!DT.dominates(getRightEntry(), Other->getRightEntry()))
    return false;

  return true;
}

RegionAnalyzer::RegionAnalyzer(BasicBlock *BB, ControlFlowGraphInfo &CFGInfo)
    : DivergentBB(BB), CFGInfo(CFGInfo) {

  BranchInst *Bi = dyn_cast<BranchInst>(DivergentBB->getTerminator());
  assert(Bi && Bi->isConditional() &&
         "Top BB needs to have a conditional branch");
  DivergentCondition = Bi->getCondition();
}

void RegionAnalyzer::computeSARegionMatch() {
  RegionMeldingProfitabilityModel ScoringFunc;
  auto SMSA =
      SmithWaterman<Region *, SmallVectorImpl<Region *>, nullptr>(ScoringFunc);

  auto RegionAlignment = SMSA.compute(LeftRegions, RightRegions);
  int AlignedReginPairs = 0;
  for (auto Entry : RegionAlignment) {
    Region *L = Entry.getLeft();
    Region *R = Entry.getRight();
    if (Entry.match()) {
      RegionComparator RC(L, R);
      bool Check = RC.compare();
      assert(Check && "Aligned regions are not similar!");
      std::shared_ptr<MergeableRegionPair> RegionPair =
          std::make_shared<MergeableRegionPair>(*L, *R, RC);

      Result.BestRegionMatch.push_back(RegionPair);
      AlignedReginPairs++;
    }
  }

  INFO << "Number of aligned region pairs : " << AlignedReginPairs << "\n";
}

void RegionAnalyzer::computeGreedyRegionMatch() {
  SmallVector<std::shared_ptr<MergeableRegionPair>, 0> MergeableRegionPairs;
  // case 3 : do a N*N region simlarty check
  DEBUG << "Performing a N*N region similarity check\n";
  for (Region *LRegion : LeftRegions) {
    for (Region *RRegion : RightRegions) {
      RegionComparator RC(LRegion, RRegion);
      if (RC.compare()) {
        std::shared_ptr<MergeableRegionPair> RegionPair =
            std::make_shared<MergeableRegionPair>(*LRegion, *RRegion, RC);

        // DEBUG << "Regions are similar\n";
        // DEBUG << *regionPair << "\n";
        // DEBUG << "Similarity score is " << regionPair->GetSimilarityScore()
        //       << "\n";

        MergeableRegionPairs.push_back(RegionPair);
      } else {
        // DEBUG << "Regions are not similar\n";
      }
    }
  }

  DominatorTree &DT = getCFGInfo().getDomTree();
  // Sort the mergeable region pairs based on their simlarity
  auto Comparator = [&](std::shared_ptr<MergeableRegionPair> &RP1,
                        std::shared_ptr<MergeableRegionPair> &RP2) {
    if (*RP1 == *RP2) {
      return RP1->dominates(RP2, DT);
    }
    return *RP1 > *RP2;
  };
  std::sort(MergeableRegionPairs.begin(), MergeableRegionPairs.end(),
            Comparator);

  // add the most profitable match first
  if (MergeableRegionPairs.size() > 0) {
    Result.BestRegionMatch.push_back(MergeableRegionPairs[0]);
  }

  // then add next best mergeable pair given that prev best dominates
  // next best in L and R paths
  for (unsigned I = 1; I < MergeableRegionPairs.size(); I++) {
    auto &MRPair = MergeableRegionPairs[I];

    auto &CurrBest = Result.BestRegionMatch.back();

    if (CurrBest->dominates(MRPair, DT))
      Result.BestRegionMatch.push_back(MRPair);
  }
}

void RegionAnalyzer::computeRegionMatch() {

  BasicBlock *LeftEntry = DivergentBB->getTerminator()->getSuccessor(0);
  BasicBlock *RightEntry = DivergentBB->getTerminator()->getSuccessor(1);

  Result.DivBlock = DivergentBB;

  // running branch fusion only
  if (RunBranchFusionOnly) {
    BasicBlock *LeftUniqueSucc = LeftEntry->getUniqueSuccessor();
    BasicBlock *RightUniqueSucc = RightEntry->getUniqueSuccessor();
    if (LeftUniqueSucc && RightUniqueSucc &&
        LeftUniqueSucc == RightUniqueSucc) {
      Result.BestBbMatchSimilarityScore =
          Utils::computeBlockSimilarity(LeftEntry, RightEntry);
      DEBUG << "Branch fusion can be applied to basic blocks ";
      LeftEntry->printAsOperand(errs(), false);
      errs() << ", ";
      RightEntry->printAsOperand(errs(), false);
      errs() << ", similarity score" << Result.BestBbMatchSimilarityScore
             << "\n";
      Result.BestBbMatch.first = LeftEntry;
      Result.BestBbMatch.second = RightEntry;
      Result.HasBlockMatch = true;
      // this does not need region replication
      Result.RequireRegionReplication = false;
    }
    return;
  }

  // find the regions on left and right paths
  findMergeableRegions(*DivergentBB);

  // Case 1 : No regions found. L and R paths have single BB
  if (LeftRegions.empty() && RightRegions.empty()) {
    Result.BestBbMatchSimilarityScore =
        Utils::computeBlockSimilarity(LeftEntry, RightEntry);
    DEBUG << "Basic blocks ";
    LeftEntry->printAsOperand(errs(), false);
    errs() << ", ";
    RightEntry->printAsOperand(errs(), false);
    errs() << " can be melded, similarity score "
           << Result.BestBbMatchSimilarityScore << "\n";

    Result.BestBbMatch.first = LeftEntry;
    Result.BestBbMatch.second = RightEntry;
    Result.HasBlockMatch = true;
    // this does not need region replication
    Result.RequireRegionReplication = false;
    return;
  }

  // Case 2 : L or R path has single BB
  if (LeftRegions.empty() || RightRegions.empty()) {

    if (LeftRegions.empty()) {
      Result.BestBbMatch.first = LeftEntry;
    }
    if (RightRegions.empty()) {
      Result.BestBbMatch.second = RightEntry;
    }

    PostDominatorTree &PDT = getCFGInfo().getPostDomTree();
    SmallVector<BasicBlock *, 8> MergeableBlocks;
    BasicBlock *IPDom = PDT.getNode(DivergentBB)->getIDom()->getBlock();

    assert(IPDom && "No IPDOM for divergent branch! This case is not handled");

    Region *ReplicatedRegion = nullptr;

    if (!Result.BestBbMatch.first) {
      assert(LeftRegions.size() > 0 && Result.BestBbMatch.second);
      findMergeableBBsInRegions(LeftEntry, LeftRegions, MergeableBlocks);
      if (MergeableBlocks.size()) {
        Result.BestBbMatch.first =
            findMostSimilarBb(Result.BestBbMatch.second, MergeableBlocks);

        for (auto *R : LeftRegions) {
          if (R->contains(Result.BestBbMatch.first)) {
            ReplicatedRegion = R;
            break;
          }
        }
      }
    }

    if (!Result.BestBbMatch.second) {
      assert(RightRegions.size() > 0 && Result.BestBbMatch.first);
      MergeableBlocks.clear();
      findMergeableBBsInRegions(RightEntry, RightRegions, MergeableBlocks);
      if (MergeableBlocks.size()) {
        Result.BestBbMatch.second =
            findMostSimilarBb(Result.BestBbMatch.first, MergeableBlocks);
        for (auto *R : RightRegions) {
          if (R->contains(Result.BestBbMatch.second)) {
            ReplicatedRegion = R;
            break;
          }
        }
      }
    }

    BasicBlock *L = Result.BestBbMatch.first;
    BasicBlock *R = Result.BestBbMatch.second;

    // if profitable match is found
    if (L && R) {
      Result.HasBlockMatch = true;

      DEBUG << "Block to region melding is possible with blocks "
            << Utils::getNameStr(L) << ", " << Utils::getNameStr(R) << "\n";

      // decide if this needs region replicatoin
      BasicBlock *LeftEntry = DivergentBB->getTerminator()->getSuccessor(0);
      BasicBlock *RightEntry = DivergentBB->getTerminator()->getSuccessor(1);
      PostDominatorTree &PDT = getCFGInfo().getPostDomTree();

      Result.RequireRegionReplication =
          !PDT.dominates(L, LeftEntry) || !PDT.dominates(R, RightEntry);

      if (Result.RequireRegionReplication) {
        // utility function to get branch cost
        auto GetBrCost = [&]() -> int {
          return getCFGInfo()
              .getTTI()
              .getCFInstrCost(Instruction::Br, TTI::TCK_CodeSize)
              .getValue()
              .getValue();
        };
        DEBUG << "Melding requires region replication\n";
        Result.BestBbMatchSimilarityScore =
            Utils::computeBlockSimilarity(L, R, ReplicatedRegion, GetBrCost);
        // BestBbMatch.first, BestBbMatch.second);

      } else {
        Result.BestBbMatchSimilarityScore = Utils::computeBlockSimilarity(L, R);
      }

      DEBUG << "Similarity score = " << Result.BestBbMatchSimilarityScore
            << "\n";
    }

    return;
  }
  if (PerformGreedyRegionMatch)
    computeGreedyRegionMatch();
  else
    computeSARegionMatch();

  DEBUG << "Region pairs in profitabilty order  \n";
  for (unsigned I = 0; I < Result.BestRegionMatch.size(); ++I) {
    auto BestPair = Result.BestRegionMatch[I];
    DEBUG << *BestPair
          << ", similarity score = " << BestPair->getSimilarityScore() << "\n";
  }
}

void RegionAnalyzer::findMergeableBBsInRegions(
    BasicBlock *From, SmallVectorImpl<Region *> &Regions,
    SmallVectorImpl<BasicBlock *> &MergeableBBs) {

  auto LI = getCFGInfo().getLoopInfo();
  auto &DT = getCFGInfo().getDomTree();
  auto &PDT = getCFGInfo().getPostDomTree();

  auto IsInsideLoop = [&](BasicBlock *BB) -> bool {
    Loop *LoopOfBB = LI->getLoopFor(BB);
    if (LoopOfBB) {
      BasicBlock *LoopHeader = LoopOfBB->getHeader();
      if (DT.dominates(From, LoopHeader))
        return true;
    }
    return false;
  };
  // any basic block containd in the regions is a valid location to merge
  // including their exits
  for (Region *R : Regions) {
    for (auto *Cand : R->blocks()) {
      // FIXME : avoid merging with basic blocks inside loops
      // if region replication is not allowed, basic blocks that post dominates
      // From can be a meld candidate
      if (!IsInsideLoop(Cand)) {
        if (DisableRegionReplication) {
          if (PDT.dominates(Cand, From))
            MergeableBBs.push_back(Cand);
        } else {
          MergeableBBs.push_back(Cand);
        }
      }
    }
    // // check region exit
    // BasicBlock* Exit = R->getExit();
    // if (!IsInsideLoop(Exit)){
    //   MergeableBBs.push_back(Exit);
    // }
  }
}

void RegionAnalyzer::findMergeableRegions(BasicBlock &BB) {
  auto RI = getCFGInfo().getRegionInfo();
  DominatorTree &DT = getCFGInfo().getDomTree();
  PostDominatorTree &PDT = getCFGInfo().getPostDomTree();

  Region *R = RI->getRegionFor(&BB);

  assert(BB.getTerminator()->getNumSuccessors() == 2 &&
         "CFMelder : Entry block must have 2 successors!");

  BasicBlock *LeftEntry = BB.getTerminator()->getSuccessor(0);
  BasicBlock *RightEntry = BB.getTerminator()->getSuccessor(1);

  SmallVector<Region *, 0> VisitedRegions;

  // get all direct children of R and split them L and R paths
  // regions are added to L  and R lists based on dominance order
  for (auto It = R->begin(); It != R->end(); It++) {
    Region &SubR = **It;

    // if sub region is not direct child of R skip
    if (SubR.getParent() != R)
      continue;

    BasicBlock *EntryBb = SubR.getEntry();

    // check if this region beglongs to left or right paths
    if (DT.dominates(LeftEntry, EntryBb) && PDT.dominates(EntryBb, LeftEntry)) {

      auto ItLR = LeftRegions.begin();
      for (; ItLR != LeftRegions.end(); ItLR++) {
        if (DT.dominates(EntryBb, (*ItLR)->getEntry()))
          break;
      }

      LeftRegions.insert(ItLR, &SubR);
    }

    if (DT.dominates(RightEntry, EntryBb) &&
        PDT.dominates(EntryBb, RightEntry)) {
      // add regions based on dominance order
      auto ItRR = RightRegions.begin();
      for (; ItRR != RightRegions.end(); ItRR++) {
        if (DT.dominates(EntryBb, (*ItRR)->getEntry()))
          break;
      }
      RightRegions.insert(ItRR, &SubR);
    }
  }

#ifdef CFMELDER_DEBUG
  DEBUG << "Left regions : ";
  printRegionList(LeftRegions, DT);
  errs() << "\n";
  DEBUG << "Right regions : ";
  printRegionList(RightRegions, DT);
  errs() << "\n";
#endif

  // verify dominance order
  assert(verifyRegionList(LeftRegions, DT) &&
         "Left regions are not in dominance order!");
  assert(verifyRegionList(RightRegions, DT) &&
         "Right regions are not in dominance order!");
}

void RegionAnalyzer::printAnalysis(llvm::raw_ostream &OS) { Result.print(OS); }

unsigned RegionAnalysisResult::regionMatchSize() const {
  if (HasBlockMatch)
    return 1;
  return BestRegionMatch.size();
}

std::pair<BasicBlock *, BasicBlock *>
RegionAnalysisResult::getRegionMatchEntryBlocks(unsigned I) {

  if (HasBlockMatch)
    return BestBbMatch;
  std::shared_ptr<MergeableRegionPair> RP = BestRegionMatch[I];

  return std::pair<BasicBlock *, BasicBlock *>(RP->getLeftEntry(),
                                               RP->getRightEntry());
}

std::pair<BasicBlock *, BasicBlock *>
RegionAnalysisResult::getRegionMatchExitBlocks(unsigned I) {
  // for single BB match exit block is null
  if (HasBlockMatch)
    return std::pair<BasicBlock *, BasicBlock *>(nullptr, nullptr);
  std::shared_ptr<MergeableRegionPair> RP = BestRegionMatch[I];

  return std::pair<BasicBlock *, BasicBlock *>(RP->getLeftExit(),
                                               RP->getRightExit());
}

DenseMap<BasicBlock *, BasicBlock *>
RegionAnalysisResult::getRegionMatch(unsigned I) {
  DenseMap<BasicBlock *, BasicBlock *> BbMap;

  if (HasBlockMatch) {
    assert(I == 0 && "only one basicblock match is expected");
    BbMap.insert(BestBbMatch);
    return BbMap;
  }

  assert(I < BestRegionMatch.size() &&
         "requested list index is greater than its size!");

  std::shared_ptr<MergeableRegionPair> RP = BestRegionMatch[I];

  SmallVector<BasicBlock *, 0> WorkList;
  SmallVector<BasicBlock *, 0> VisitedBBs;
  WorkList.push_back(RP->getLeftEntry());

  while (!WorkList.empty()) {
    BasicBlock *CurrLeftBb = WorkList.pop_back_val();
    BasicBlock *MatchingRightBb = RP->getMatchingRightBb(CurrLeftBb);

    // add to map
    BbMap.insert(
        std::pair<BasicBlock *, BasicBlock *>(CurrLeftBb, MatchingRightBb));

    // this block is visited
    VisitedBBs.push_back(CurrLeftBb);
    // add all in-region successors to worklist
    for (unsigned I = 0; I < CurrLeftBb->getTerminator()->getNumSuccessors();
         I++) {
      BasicBlock *Succ = CurrLeftBb->getTerminator()->getSuccessor(I);
      if (Succ != RP->getLeftExit() &&
          std::find(VisitedBBs.begin(), VisitedBBs.end(), Succ) ==
              VisitedBBs.end())
        WorkList.push_back(Succ);
    }
  }

  // add matching exit blocks
  BbMap.insert(std::pair<BasicBlock *, BasicBlock *>(RP->getLeftExit(),
                                                     RP->getRightExit()));

  return BbMap;
}

unsigned RegionAnalysisResult::getMostProfitableRegionMatchIndex() {
  if (HasBlockMatch)
    return 0;

  double MaxProfit = 0.0;
  unsigned MaxIndex = 0;
  for (unsigned I = 0; I < BestRegionMatch.size(); I++) {
    if (BestRegionMatch[I]->getSimilarityScore() > MaxProfit) {
      MaxIndex = I;
      MaxProfit = BestRegionMatch[I]->getSimilarityScore();
    }
  }
  return MaxIndex;
}

bool RegionAnalysisResult::hasAnyProfitableMatch() {
  if (HasBlockMatch) {
    // if merging 2 basic blocks, avoid merging single branch blocks (size > 1)
    return BestBbMatchSimilarityScore >= SimilarityThreshold &&
           (BestBbMatch.first->size() > 1 && BestBbMatch.second->size() > 1);
  }

  if (BestRegionMatch.size() > 0) {
    for (auto &RegionPair : BestRegionMatch) {
      if (RegionPair->getSimilarityScore() >= SimilarityThreshold)
        return true;
    }
  }
  return false;
}

bool RegionAnalysisResult::isRegionMatchProfitable(unsigned Index) {
  assert(Index < regionMatchSize() && "Region match index out of bounds!");
  if (HasBlockMatch) {
    return hasAnyProfitableMatch();
  }
  return BestRegionMatch[Index]->getSimilarityScore() >= SimilarityThreshold;
}

void RegionAnalysisResult::print(llvm::raw_ostream &OS) {
  if (HasBlockMatch) {
    OS << "Similarity Score : " << BestBbMatchSimilarityScore << "\n";
    OS << "Merge at BB level : \n"
       << "   merging BB ";
    BestBbMatch.first->printAsOperand(OS, false);
    OS << " with  ";
    BestBbMatch.second->printAsOperand(errs(), false);
    OS << "   requires " << (requireRegionReplication() ? "" : " no ")
       << "region replication\n";

  } else if (BestRegionMatch.size() > 0) {
    OS << "Merge at REGION level : \n";
    for (unsigned I = 0; I < BestRegionMatch.size(); I++) {
      OS << "Index : " << I << "\n";
      OS << "Similarity Score : " << BestRegionMatch[I]->getSimilarityScore()
         << "\n"
         << "   merging region entry ";
      BestRegionMatch[I]->getLeftEntry()->printAsOperand(errs(), false);
      OS << " with "
         << " region entry ";
      BestRegionMatch[I]->getRightEntry()->printAsOperand(errs(), false);
      OS << "\n"
         << "   merging region exit ";
      BestRegionMatch[I]->getLeftExit()->printAsOperand(errs(), false);
      OS << " with "
         << " region exit ";
      BestRegionMatch[I]->getRightExit()->printAsOperand(errs(), false);
      OS << "\n";
    }
  }
}

Value *RegionAnalysisResult::getDivCondition() {
  BranchInst *BI = dyn_cast<BranchInst>(DivBlock->getTerminator());
  assert(BI->isConditional() &&
         "No conditional branch at the end of divergent block!");
  return BI->getCondition();
}
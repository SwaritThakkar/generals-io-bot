#!/usr/bin/env bash
# Re-download any paper that was removed from papers/ on 2026-08-28.
# Nothing was lost: every source URL is here. Run this from papers/.
#
#   bash restore-removed.sh            # restore everything
#   bash restore-removed.sh puppet     # restore just the ones matching "puppet"
set -u
filter="${1:-}"

get() {  # get <url> <filename>
  [ -n "$filter" ] && [[ "$2" != *"$filter"* ]] && return 0
  code=$(curl -sL --max-time 90 -o "$2" -w "%{http_code}" "$1")
  if [ "$code" = "200" ] && file "$2" | grep -q PDF; then
    printf "  ok    %s\n" "$2"
  else
    printf "  FAIL  %-58s http %s\n" "$2" "$code"; rm -f "$2"
  fi
}

B=https://skatgame.net/mburo/ps
get "$B/aiide15-puppet.pdf"                  "barriga2015_puppet_search_AIIDE.pdf"
get "$B/aaai12-sbs.pdf"                      "saffidine2012_alphabeta_simultaneous_moves_AAAI.pdf"
get "$B/aiide12-combat.pdf"                  "churchill2012_ABCD_rts_combat_search_AIIDE.pdf"
get "$B/combat13.pdf"                        "churchill2013_portfolio_greedy_search_CIG.pdf"
get "$B/HierarchicalSearch-AIIDE-2014.pdf"   "stanescu2014_hierarchical_adversarial_search_AIIDE.pdf"
get "$B/mcplan.pdf"                          "chung2005_monte_carlo_planning_rts_CIG.pdf"
get "$B/planprop.pdf"                        "buro2006_build_order_optimization_first_look.pdf"

C=https://davechurchill.ca/publications/pdf
get "$C/aiide11-bo.pdf"                      "churchill2011_build_order_optimization_DFBB_AIIDE.pdf"
get "$C/aiide15_churchill_prismata.pdf"      "churchill2015_hierarchical_portfolio_search_AIIDE.pdf"
get "$C/cog20influence.pdf"                  "critch2020_influence_maps_heuristic_search_sneak_CoG.pdf"
get "$C/aiide12ws-search.pdf"                "churchill2012_incorporating_search_into_rts_agents.pdf"
get "$C/starcraft_survey.pdf"                "ontanon2013_survey_rts_game_ai_TCIAIG.pdf"

H=http://harabor.net/data/papers
get "$H/harabor-grastien-aaai11.pdf"         "harabor2011_jump_point_search_AAAI.pdf"
get "$H/harabor-grastien-icaps14.pdf"        "harabor2014_improving_jump_point_search_ICAPS.pdf"

I=https://idm-lab.org/bib/abstracts/papers
get "$I/aamas06.pdf"                         "koenig2006_real_time_adaptive_astar_RTAA_AAMAS.pdf"
get "$I/jaamas09.pdf"                        "koenig2009_comparing_realtime_incremental_heuristic_search_JAAMAS.pdf"
get "$I/aij01.pdf"                           "koenig2001_minimax_realtime_heuristic_search_AIJ.pdf"
get "$I/aamas05b.pdf"                        "koenig2005_adaptive_astar_AAMAS.pdf"

N=http://www.cs.ualberta.ca/~nathanst/papers
get "$N/sharon2015cbsjournal.pdf"            "sharon2015_conflict_based_search_MAPF_AIJ.pdf"
get "$N/sharon12cbs.pdf"                     "sharon2012_conflict_based_search_MAPF_AAAI.pdf"
get "$N/PRLRTS.pdf"                          "bulitko2007_graph_abstraction_realtime_heuristic_search_JAIR.pdf"
get "$N/partialpathfinding.pdf"              "sturtevant2005_partial_refinement_astar_abstraction_AAAI.pdf"
get "$N/CPRA.pdf"                            "sturtevant2006_collaborative_pathfinding_map_abstraction_AIIDE.pdf"
get "$N/stern-mapf.pdf"                      "stern2019_mapf_definitions_variants_benchmarks_SoCS.pdf"

get "https://arxiv.org/pdf/1702.03401"                                    "plaat1996_mtdf_best_first_fixed_depth_minimax_AIJ.pdf"
get "https://arxiv.org/pdf/1304.1133"                                     "russell1990_finegrained_decision_theoretic_search_control_UAI.pdf"
get "https://dke.maastrichtuniversity.nl/m.winands/documents/pnchapter.pdf" "winands_proof_number_search_and_variants.pdf"
get "https://webdocs.cs.ualberta.ca/~mmueller/ps/ICGA2012PNS.pdf"         "kishimoto2012_game_tree_search_using_proof_numbers_ICGA.pdf"

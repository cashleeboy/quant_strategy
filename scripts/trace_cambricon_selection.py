#!/usr/bin/env python3
"""Read-only reconstruction of frozen broad fixed15 decisions, not a new strategy."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from analysis.single_account_adaptive import family_score, prior_weights, history_risk, constrained_selection
from factors.preprocess import cross_sectional_zscore
from strategies.holding_buffer import ordered_scores
from scripts.run_six_strategy_ic_comparison import load_prices


def trace(scores, industry, held, corr, spec, focus):
    ranked = ordered_scores(scores).head(spec['candidate_limit'])
    retained = [s for s in ranked.head(30).index if s in held]
    order = retained + [s for s in ranked.index if s not in retained]
    chosen, counts, tech_count, records = [], {}, 0, []
    industry = industry.fillna('UNKNOWN').replace('', 'UNKNOWN')
    for symbol in order:
        sector = industry.get(symbol, 'UNKNOWN')
        tech = sector in spec['theme_industries']
        reason, blockers = 'selected', []
        if len(chosen) == 15:
            reason = 'slots_already_full'
        elif counts.get(sector, 0) >= int(np.floor(15 * spec['industry_cap'] + 1e-9)):
            reason = 'industry_cap'
            blockers = [s for s in chosen if industry.get(s, 'UNKNOWN') == sector]
        elif tech and tech_count >= int(np.floor(15 * spec['theme_cap'] + 1e-9)):
            reason = 'technology_cap'
            blockers = [s for s in chosen if industry.get(s, 'UNKNOWN') in spec['theme_industries']]
        elif symbol not in corr.index or not np.isfinite(corr.loc[symbol, symbol]):
            reason = 'missing_risk_history'
        elif chosen:
            pairs = corr.loc[symbol, chosen]
            blockers = pairs.index[pairs.isna() | (pairs > spec['correlation_cap'])].tolist()
            if blockers:
                reason = 'correlation_or_missing_pair'
        records.append(dict(symbol=symbol, reason=reason, blockers=';'.join(blockers),
                            retained=symbol in retained, slots_before=len(chosen),
                            tech_before=tech_count, industry_before=counts.get(sector, 0)))
        if reason == 'selected':
            chosen.append(symbol)
            counts[sector] = counts.get(sector, 0) + 1
            tech_count += int(tech)
    assert chosen == constrained_selection(scores, industry, held, 15, corr, spec)
    result = next((r for r in records if r['symbol'] == focus),
                  dict(symbol=focus, reason='outside_top100', blockers='', retained=False))
    return chosen, result


def main():
    p = json.loads((ROOT/'research_protocols/20261020_single_account_adaptive_v1.json').read_text())
    families = json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
    frozen = ROOT/'output/candidate_factors_20261022'
    account = ROOT/'output/fixed15_ml_delisting_v4_20261024/BASE_BROAD_3bp'
    out = ROOT/'output/cambricon_trace_20261024'
    out.mkdir(exist_ok=True)
    focus = '688256.SH'
    weights = pd.read_csv(frozen/'factor_weights.csv', parse_dates=['date'])
    weights = weights[weights.arm == 'BASE'].pivot(index='date', columns='factor', values='weight').sort_index()
    rankings = pd.read_csv(frozen/'rankings.csv.gz', parse_dates=['date'])
    rankings = rankings[rankings.arm == 'BASE']
    positions = pd.read_csv(account/'positions.csv.gz', parse_dates=['date'])
    targets = pd.read_csv(account/'targets.csv.gz', parse_dates=['date','execution_date'])
    decisions = pd.read_csv(account/'decisions.csv.gz', parse_dates=['date']).set_index('date')
    executions = pd.read_csv(account/'executions.csv.gz', parse_dates=['date','signal_date']).set_index('signal_date')
    prices = load_prices(p)
    returns = prices.pct_change(fill_method=None)
    weeks = pd.DatetimeIndex(json.loads((frozen/'schedules.json').read_text())['WEEKLY'])
    rows, factor_rows, family_rows = [], [], []
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_202[56]*')):
        panel = pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz', parse_dates=['date']).set_index(['date','symbol'])
        for date, g_multi in panel.groupby(level=0):
            if date not in weeks or date not in decisions.index:
                continue
            g = g_multi.droplevel(0)
            updated = weights.index[weights.index <= date][-1]
            w = weights.loc[updated]
            pool = family_score(g_multi, families, prior_weights(families)).droplevel(0).index
            score = ordered_scores(family_score(g_multi, families, w, require_all=False).droplevel(0).reindex(pool))
            saved = rankings[rankings.date == date].sort_values('rank').set_index('symbol').score
            assert saved.index.tolist() == score.head(100).index.tolist()
            np.testing.assert_allclose(saved, score.head(100), atol=1e-12)
            held = set(positions.loc[(positions.date == date) & (positions.units > 0), 'symbol'])
            corr, _ = history_risk(returns, date, saved.index, p['selection'])
            chosen, reason = trace(saved, g.industry, held, corr, p['selection'], focus)
            chosen_no_buffer, no_buffer_reason = trace(saved, g.industry, set(), corr, p['selection'], focus)
            expected = targets.loc[(targets.date == date) & (targets.weight > 0), 'symbol'].tolist()
            assert set(chosen) == set(expected), f'Target mismatch {date}'
            chosen_ranks = [int(saved.index.get_loc(s) + 1) for s in chosen]
            row = dict(date=date, execution_date=executions.loc[date, 'date'],
                       full_rank=(score.index.get_loc(focus) + 1 if focus in score.index else np.nan),
                       score=score.get(focus, np.nan), eligible=focus in score.index,
                       held_before=focus in held, chosen=focus in chosen,
                       chosen_without_buffer=focus in chosen_no_buffer,
                       no_buffer_reason=no_buffer_reason['reason'],
                       retained_selected=sum(s in held and saved.index.get_loc(s) < 30 for s in chosen),
                       chosen_rank_min=min(chosen_ranks), chosen_rank_max=max(chosen_ranks),
                       target_weight=float(targets.loc[(targets.date == date)&(targets.symbol == focus),'weight'].sum()),
                       budget=decisions.loc[date, 'budget'], weight_update=updated, **reason)
            rows.append(row)
            if focus not in g.index:
                continue
            composites, budgets = {}, {}
            for name, cols in families.items():
                fw = w.reindex(cols)
                budget = fw.sum()
                if budget <= 1e-14:
                    continue
                denom = g_multi[cols].notna().astype(float).dot(fw)
                composites[name] = g_multi[cols].mul(fw).sum(axis=1, min_count=1)/denom.where(denom > 0)
                budgets[name] = budget
                for c in cols:
                    factor_rows.append(dict(date=date, factor=c, family=name,
                                            standardized_factor=g.loc[focus,c], nominal_weight=w[c]))
            z = cross_sectional_zscore(pd.DataFrame(composites))
            focus_key = (date, focus)
            denom = sum(budgets[n] for n in budgets if pd.notna(z.loc[focus_key,n]))
            for name, budget in budgets.items():
                family_rows.append(dict(date=date, family=name, family_z=z.loc[focus_key,name],
                                        budget=budget, score_contribution=z.loc[focus_key,name]*budget/denom))
            if focus in score.index:
                np.testing.assert_allclose(sum(r['score_contribution'] for r in family_rows if r['date']==date and pd.notna(r['score_contribution'])),score[focus],atol=1e-12)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'weekly_trace.csv',index=False)
    pd.DataFrame(factor_rows).to_csv(out/'factor_trace.csv',index=False)
    pd.DataFrame(family_rows).to_csv(out/'family_trace.csv',index=False)
    trades = pd.read_csv(account/'trades.csv.gz',parse_dates=['date'])
    trades = trades[trades.symbol == focus]
    trades.to_csv(out/'trades.csv',index=False)
    print(frame.to_string(index=False))
    print('PASS: all reconstructed top100 scores, selected portfolios and family contributions match frozen outputs;',len(frame),'weekly decisions')


if __name__ == '__main__':
    main()

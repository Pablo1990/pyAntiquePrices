"""Sub-commands: ``links``, ``deal``, ``ledger`` and ``template``.

    pyantique-prices links --object "pocket watch" --maker Omega --material silver
    pyantique-prices deal --asking 120 --p25 150 --p50 220 --p75 300 --shipping 8
    pyantique-prices ledger add "Omega watch" --price 120 --costs 8 --where "flea market"
    pyantique-prices ledger sell 3 --price 260 --fees 30
    pyantique-prices ledger summary
    pyantique-prices template > my_sales.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMANDS = {"links", "deal", "ledger", "template"}

SALES_TEMPLATE_HEADER = (
    "title,description,object_type,manufacturer,artist,period,material,condition,country,"
    "auction_house,sale_date,currency,hammer_price,final_price,buyer_premium,price_basis,source_url"
)
SALES_TEMPLATE_ROW = (
    '"Omega silver hunter pocket watch","Swiss, working, minor wear","pocket watch","Omega",,'
    '"c. 1910","silver","good","Switzerland","My own sale",2026-03-14,EUR,,260,,realized,'
    "my-own-sale-001"
)


def _identification_from_args(args) -> dict:
    def cand(name):
        return [{"name": name, "confidence": 0.9}] if name else []

    return {
        "object_type": args.object,
        "subtype": args.subtype,
        "period": args.period,
        "likely_period": args.period,
        "materials": [args.material] if args.material else [],
        "manufacturer_candidates": cand(args.maker),
        "artist_candidates": cand(args.artist),
    }


def _cmd_links(args) -> int:
    from .config import Settings
    from .lookup import build_lookup_links, format_lookup_links

    block = build_lookup_links(
        _identification_from_args(args),
        f"Extra keywords: {args.keywords}" if args.keywords else "",
        ebay_domain=args.ebay_domain or Settings().ebay_domain,
    )
    if not block["links"]:
        print("Give at least --object, --maker, --artist or --keywords.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(block, indent=2, ensure_ascii=False))
    else:
        print("\n".join(format_lookup_links(block)))
    return 0


def _valuation_from_args(args) -> dict | None:
    if args.appraisal:
        data = json.loads(Path(args.appraisal).read_text(encoding="utf-8"))
        return data.get("valuation") or data
    if args.p50 is None:
        return None
    return {
        "p25": args.p25 if args.p25 is not None else args.p50 * 0.7,
        "p50": args.p50,
        "p75": args.p75 if args.p75 is not None else args.p50 * 1.4,
        "num_comparables": args.n_comparables,
        "effective_comparables": args.n_comparables,
        "valuation_available": True,
    }


def _cmd_deal(args) -> int:
    from .deals import assess_deal, format_deal

    valuation = _valuation_from_args(args)
    if valuation is None:
        print("Give --appraisal FILE.json (from --json-out) or at least --p50.", file=sys.stderr)
        return 1
    factor = 1.0
    if args.use_ledger:
        factor = _ledger_factor()
    deal = assess_deal(
        valuation,
        asking_price=args.asking,
        shipping=args.shipping,
        buyer_premium_pct=args.buyer_premium,
        vat_pct=args.vat,
        restoration=args.restoration,
        other_costs=args.other_costs,
        for_resale=not args.keep,
        resale_fee_pct=args.resale_fee,
        resale_shipping=args.resale_shipping,
        min_margin=args.min_margin,
        calibration_factor=factor,
    )
    print(json.dumps(deal, indent=2) if args.json else "\n".join(format_deal(deal, args.currency)))
    return 0 if deal["verdict"] != "no_verdict" else 1


def _session_factory():
    from .config import Settings
    from .data.database import create_tables, get_engine, get_session_factory

    engine = get_engine(Settings().database_url)
    create_tables(engine)
    return get_session_factory(engine)


def _ledger_factor() -> float:
    from .ledger import calibration_factor

    with _session_factory()() as session:
        return calibration_factor(session)


def _fmt_item(item) -> str:
    from .ledger.service import cost_basis, profit

    cost = cost_basis(item)
    bits = [f"#{item.id:<3} {item.status:<5} {item.title[:38]:<38} paid {cost:>8.2f} {item.currency or ''}"]
    if item.status == "sold":
        bits.append(f"sold {item.sold_price:>8.2f} (profit {profit(item):+.2f})")
    elif item.estimate_mid:
        bits.append(f"est. ~{item.estimate_mid:.0f}")
    return " | ".join(bits)


def _cmd_ledger(args) -> int:
    from . import ledger

    with _session_factory()() as session:
        try:
            if args.action == "add":
                estimate = {}
                identification = None
                if args.appraisal:
                    data = json.loads(Path(args.appraisal).read_text(encoding="utf-8"))
                    estimate = data.get("valuation") or {}
                    identification = data.get("identification")
                elif args.est_mid is not None:
                    estimate = {"p25": args.est_low, "p50": args.est_mid, "p75": args.est_high}
                item = ledger.add_purchase(
                    session, title=args.title, price=args.price, costs=args.costs,
                    date=args.date, where=args.where, currency=args.currency,
                    identification=identification, valuation=estimate or None,
                    deal=json.loads(Path(args.appraisal).read_text(encoding="utf-8")).get("deal")
                    if args.appraisal else None,
                    notes=args.notes, object_type=args.object_type, source_url=args.url,
                )
                print(f"Recorded purchase #{item.id}: {item.title}")
            elif args.action == "sell":
                item = ledger.record_sale(session, args.id, price=args.price, fees=args.fees,
                                          date=args.date, where=args.where)
                from .ledger.service import profit
                print(f"Recorded sale of #{item.id}: profit {profit(item):+.2f} {item.currency}")
            elif args.action == "list":
                items = ledger.list_items(session, args.status)
                print("\n".join(_fmt_item(i) for i in items) or "Ledger is empty.")
            elif args.action == "summary":
                print(json.dumps(ledger.summarize(session), indent=2))
            elif args.action == "calibration":
                print(f"{ledger.calibration_factor(session):.3f}")
            elif args.action == "export":
                n = ledger.export_sales_csv(session, args.path)
                print(f"Wrote {n} sold item(s) to {args.path}")
            elif args.action == "sync":
                from .config import Settings
                print(f"Added {ledger.sync_to_sales(session, Settings().base_currency)} sale(s) to comparables.")
        except (ValueError, KeyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


def _cmd_template(_args) -> int:
    print(SALES_TEMPLATE_HEADER)
    print(SALES_TEMPLATE_ROW)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyantique-prices", description="Tools for assessing antiques and art.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("links", help="Search links to check prices yourself (nothing is fetched).")
    p.add_argument("--object"); p.add_argument("--subtype"); p.add_argument("--maker")
    p.add_argument("--artist"); p.add_argument("--material"); p.add_argument("--period")
    p.add_argument("--keywords", default=""); p.add_argument("--ebay-domain")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_links)

    p = sub.add_parser("deal", help="Is this asking price worth it?")
    p.add_argument("--asking", type=float, required=True)
    p.add_argument("--appraisal", help="JSON written by an appraisal run with --json-out")
    p.add_argument("--p25", type=float); p.add_argument("--p50", type=float); p.add_argument("--p75", type=float)
    p.add_argument("--n-comparables", type=int, default=6)
    p.add_argument("--shipping", type=float, default=0.0)
    p.add_argument("--buyer-premium", type=float, default=0.0, help="Percent")
    p.add_argument("--vat", type=float, default=0.0, help="Percent")
    p.add_argument("--restoration", type=float, default=0.0)
    p.add_argument("--other-costs", type=float, default=0.0)
    p.add_argument("--keep", action="store_true", help="Buying to keep: ignore selling costs")
    p.add_argument("--resale-fee", type=float, default=0.0, help="Percent of the sale price")
    p.add_argument("--resale-shipping", type=float, default=0.0)
    p.add_argument("--min-margin", type=float, default=0.30)
    p.add_argument("--use-ledger", action="store_true", help="Correct estimates with your past sales")
    p.add_argument("--currency", default="EUR"); p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_deal)

    p = sub.add_parser("ledger", help="Track what you bought and sold.")
    lsub = p.add_subparsers(dest="action", required=True)
    a = lsub.add_parser("add"); a.add_argument("title"); a.add_argument("--price", type=float, required=True)
    a.add_argument("--costs", type=float, default=0.0); a.add_argument("--date"); a.add_argument("--where")
    a.add_argument("--currency", default="EUR"); a.add_argument("--notes"); a.add_argument("--object-type")
    a.add_argument("--url"); a.add_argument("--appraisal", help="JSON from an appraisal (--json-out)")
    a.add_argument("--est-low", type=float); a.add_argument("--est-mid", type=float); a.add_argument("--est-high", type=float)
    s = lsub.add_parser("sell"); s.add_argument("id", type=int); s.add_argument("--price", type=float, required=True)
    s.add_argument("--fees", type=float, default=0.0); s.add_argument("--date"); s.add_argument("--where")
    ls = lsub.add_parser("list"); ls.add_argument("--status", choices=("held", "sold", "kept"))
    lsub.add_parser("summary"); lsub.add_parser("calibration")
    e = lsub.add_parser("export"); e.add_argument("path")
    lsub.add_parser("sync")
    p.set_defaults(func=_cmd_ledger)

    p = sub.add_parser("template", help="Print a CSV template for your own sales data.")
    p.set_defaults(func=_cmd_template)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

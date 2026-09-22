from pathlib import Path
import json
import pandas as pd

EXPECTED = json.loads((Path(__file__).parent / 'EXPECTED_VALIDATION.json').read_text())

def load_orders(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {'.xls', '.xlsx'}:
        return pd.read_excel(p, sheet_name='Orders')
    return pd.read_csv(p, encoding='utf-8', encoding_errors='replace')

def main(path: str):
    df = load_orders(path)
    actual = {
        'rows': int(len(df)),
        'distinct_orders': int(df['Order ID'].nunique()),
        'customers': int(df['Customer ID'].nunique()),
        'years': sorted(pd.to_datetime(df['Order Date']).dt.year.unique().tolist()),
        'regions': int(df['Region'].nunique()),
        'states': int(df['State'].nunique()),
        'categories': int(df['Category'].nunique()),
        'sub_categories': int(df['Sub-Category'].nunique()),
        'sales': round(float(df['Sales'].sum()), 2),
        'profit': round(float(df['Profit'].sum()), 2),
    }
    actual['profit_ratio_pct'] = round(actual['profit'] / actual['sales'] * 100, 2)
    actual['average_order_value'] = round(actual['sales'] / actual['distinct_orders'], 2)
    print(json.dumps(actual, indent=2))
    mismatches = {k: (EXPECTED[k], actual.get(k)) for k in EXPECTED if actual.get(k) != EXPECTED[k]}
    if mismatches:
        raise SystemExit(f'Validation mismatch: {mismatches}')
    print('Validation passed.')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python validate_superstore.py <Sample - Superstore.xls|csv>')
    main(sys.argv[1])

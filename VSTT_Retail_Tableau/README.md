# VSTT — Retail Analytics in Tableau

## Project Aim

Use Tableau to turn retail transactions into management-focused views of sales growth, profitability, product performance, customer segments, geography, discounting and operational context.

## Dataset
Tableau Sample — Superstore (Orders table), containing 9,994 line items from 2014–2017. Tableau provides this fictitious retail dataset for learning and product demonstrations.

- [Official Tableau tutorial and source guidance](https://help.tableau.com/current/guides/get-started-tutorial/en-gb/get-started-tutorial-connect.htm)
- [Official Sample — Superstore workbook](https://www.tableau.com/sites/default/files/2021-05/Sample%20-%20Superstore.xls)

## Analytical Deliverables

1. Executive Overview Dashboard
2. Customer and Product Performance Dashboard
3. Regional or Operational Insights Dashboard
4. Tableau Story with 5–8 story points

## Repository Contents

- [Submitted assignment report](report/Rio_Roy_VSTT_Assignment_FINAL.pdf)
- [Dashboard build guide](tableau/BUILD_GUIDE.md)
- [Calculated fields reference](tableau/CALCULATED_FIELDS.md)
- [Tableau story-point plan](tableau/STORY_POINTS.md)
- [Expected validation values](tableau/EXPECTED_VALIDATION.json)
- [Superstore validation script](tableau/validate_superstore.py)

## Reproducing the Analysis

Download the official Sample — Superstore workbook, then follow the build guide to recreate the calculated fields, filters, parameters, hierarchies, dashboards, and story. Use `tableau/validate_superstore.py` and `tableau/EXPECTED_VALIDATION.json` to compare the source data against the expected control totals.

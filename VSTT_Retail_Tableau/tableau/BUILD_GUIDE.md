# Tableau Workbook Build Guide

## 1. Connect and validate
Connect to `Sample - Superstore.xls` and drag the `Orders` table onto the data canvas.
Confirm the control totals used in the report:
- 9,994 rows
- 5,009 distinct orders
- 793 customers
- Total Sales: $2,297,200.86
- Total Profit: $286,397.02
- Overall Profit Ratio: 12.47%

## 2. Set field roles
Dimensions: Order Date, Ship Date, Ship Mode, Segment, Region, State, City, Category, Sub-Category, Product Name, Order ID, Customer ID.
Measures: Sales, Profit, Quantity, Discount.
Use Postal Code only for geography.

## 3. Create hierarchies
- Time: Year > Quarter > Month > Order Date
- Product: Category > Sub-Category > Product Name
- Geography: Country > Region > State > City

## 4. Create calculated fields and parameters
Use `CALCULATED_FIELDS.md`.

## 5. Worksheets
Create at minimum:
1. KPI — Total Sales
2. KPI — Total Profit
3. KPI — Profit Ratio
4. KPI — Distinct Orders
5. KPI — Average Order Value
6. Sales and Profit by Month
7. Sales and Profit by Year
8. Category Profitability
9. Sub-Category Profitability
10. Top-N Products
11. Segment Performance
12. Top-N Customers
13. Sales and Profit by Region
14. State Profit Map
15. Discount vs Profit scatter with linear trend line
16. Shipping Days by Ship Mode
17. Loss-making States/Sub-Categories table

## 6. Dashboard 1 — Executive Overview
KPI strip at top. Add monthly sales/profit trend, yearly comparison and category summary. Add filters for Year, Region, Segment and Category.

## 7. Dashboard 2 — Customer and Product Performance
Include Segment Performance, Top-N Customers, Category/Sub-Category profitability and Top-N Products. Use the Top N parameter and Metric Selector. Add dashboard filter actions.

## 8. Dashboard 3 — Regional and Operational Insights
Include map, region comparison, loss-making locations, Discount vs Profit scatter and Shipping Days by Ship Mode. Use map selections to filter supporting charts.

## 9. Tableau Story
Create 6 story points:
1. Overall performance
2. Time trend and seasonality
3. Product/category profitability
4. Customer segment and Top-N contribution
5. Regional/discount/shipping risks
6. Management recommendations

## 10. Submission evidence
Export screenshots of the worksheets, all three dashboards and Story points. Package the workbook as `.twbx` or publish it to Tableau Public, then place the link/file with the report.

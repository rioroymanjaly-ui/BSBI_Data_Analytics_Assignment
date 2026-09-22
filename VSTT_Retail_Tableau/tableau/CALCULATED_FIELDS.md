# Tableau Calculated Fields

## Profit Ratio
```tableau
SUM([Profit]) / SUM([Sales])
```

## Average Order Value
```tableau
SUM([Sales]) / COUNTD([Order ID])
```

## Shipping Days
```tableau
DATEDIFF('day', [Order Date], [Ship Date])
```

## Loss Flag
```tableau
IF SUM([Profit]) < 0 THEN "Loss" ELSE "Profit" END
```

## Discount Band
```tableau
IF [Discount] = 0 THEN "0%"
ELSEIF [Discount] <= 0.10 THEN "Low (>0–10%)"
ELSEIF [Discount] <= 0.30 THEN "Medium (>10–30%)"
ELSE "High (>30%)"
END
```

## Metric Selector parameter
Allowable values: `Sales`, `Profit`, `Profit Ratio`.

## Selected Metric
```tableau
CASE [Metric Selector]
WHEN "Sales" THEN SUM([Sales])
WHEN "Profit" THEN SUM([Profit])
WHEN "Profit Ratio" THEN SUM([Profit]) / SUM([Sales])
END
```

## Top N parameter
Integer parameter. Suggested default: 10.
Use with a Top-N set on Product Name or Customer Name.

SELECT CompanyInfo.Ticker, DailyValues.Ticker AS Expr1
FROM   CompanyInfo CROSS JOIN
             DailyValues
/****** Select all weekly spot rates for Benzene from ZEMA  ******/
SELECT 
	   [source]
      ,[report]
      ,[numOfRecords]
      ,[date]
      ,[CONCEPT]
      ,[GEOGRAPHY]
      ,[GRADE]
      ,[PRICE]
      ,[PRICEID]
      ,[PRODUCT]
      ,[PRODUCTCODE]
      ,[TERMS]
      ,[UNIT]
      ,[InsertedDate]
  FROM [Source].[Zema].[vtIHSChemicalPriceWsWly]
  WHERE [CONCEPT] like '%HIGH%'
  ORDER BY [date] ASC
/****** Script for SelectTopNRows command from SSMS  ******/
SELECT 
	[source]
      ,[report]
      ,[numOfRecords]
      ,[date]
      ,[PRICESERIESNUMBER]
      ,[QUOTETYPE]
      ,[PRODUCT]
      ,[CURRENCYUNIT]
      ,[DESCRIPTION2]
      ,[LOW]
      ,[HIGH]
      ,[InsertedDate]
  FROM [Source].[Zema].[vtICISReports]
  /*WHERE [PRODUCT] = 'BENZENE' 
    AND [CURRENCYUNIT] = 'USD/US gal'
  ORDER BY [date] asc*/
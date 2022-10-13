/****** Script for SelectTopNRows command from SSMS  ******/
SELECT [MANDT]
      ,[MATNR]
      ,[SPRAS]
      ,[MAKTX]
      ,[MAKTG]
  FROM [Source].[SapEcc].[vtMakt]
  --WHERE [MATNR] = '000000000001037942'
  --WHERE [MATNR] like '%2071069'
  --WHERE MAKTX like 'BENZENE%' AND [SPRAS] = 'E'
  --WHERE [MATNR] in ('000000000000001056', '000000000000016113', '000000000002099002', '000000000000014689', '000000000002071069') AND [SPRAS] = 'E'
  WHERE [MATNR] in (
					SELECT DISTINCT ([MATNR])
					FROM [Source].[SapEcc].[vtEban]
					WHERE ([TXZ01] like '%BENZENE%' 
						or [TXZ01] like '%Benzene%' 
						or [TXZ01] like '%benzene%')
						and [MATNR] like '0%'
  )


  
/****** Script for SelectTopNRows command from SSMS  ******/
SELECT 
      [EKBE.EBELN] as 'Purchase Document Num'
      ,[EKBE.EBELP] as 'Line Item Num'
      ,[EKBE.VGABE] as 'Transaction Type'
      ,[EKBE.GJAHR] as 'Mat Doc Year'
      ,[EKBE.BELNR] as 'Mat Doc Num'
      ,[EKBE.BUZEI] as 'Mat Doc Line Num'
      ,[EKBE.BUDAT] as 'Doc Posting Date'
      ,[EKBE.MENGE] as 'Quantity (EKBE)'
      ,[EKBE.BPMNG] as 'QTY in PO Price Unit (EKBE)'
      ,[EKBE.DMBTR] as 'AMT in Local Curr'
      ,[EKBE.AREWR] as 'GR Value in Local Curr'
      ,[EKBE.ELIKZ] as 'Delivery Complete Ind'
      ,[EKBE.GRUND] as 'Reason for Movement'
      ,[EKBE.MATNR] as 'Mat Number'
      ,[EKBE.WERKS] as 'Plant'
      ,[EKBE.LSMNG] as 'QTY in UoM from Delivery Note'
      ,[EKBE.LSMEH] as 'UoM from Delivery Note'
      ,[EKBE.AREWW] as 'Clearing value on GR Transact Curr'

      ,[EKKO.BUKRS] as 'Company Code'
      ,[EKKO.BSTYP] as 'Document Category'
      --,[EKKO.BSART]
      --,[EKKO.BSAKZ]
      --,[EKKO.STATU]
      --,[EKKO.AEDAT]
      --,[EKKO.ERNAM]
      --,[EKKO.PINCR]
      --,[EKKO.LPONR]
      --,[EKKO.LIFNR] as 'Vendor Number'
      --,[EKKO.SPRAS]
      --,[EKKO.ZTERM]
      --,[EKKO.ZBD1T]
      --,[EKKO.ZBD2T]
      --,[EKKO.ZBD3T]
      --,[EKKO.ZBD1P]
      --,[EKKO.ZBD2P]
      ,[EKKO.EKORG] as 'Purchasing Org'
      ,[EKKO.EKGRP] as 'Purchasing Group'
      --,[EKKO.WAERS]
      --,[EKKO.WKURS]
      --,[EKKO.KUFIX]
      ,[EKKO.BEDAT] as 'Purchasing Doc Date'
      /*,[EKKO.KDATB]
      ,[EKKO.KDATE]
      ,[EKKO.BWBDT]
      ,[EKKO.ANGDT]
      ,[EKKO.BNDDT]
      ,[EKKO.GWLDT]
      ,[EKKO.IHRAN]
      ,[EKKO.VERKF]
      ,[EKKO.TELF1]
      ,[EKKO.RESWK]
      ,[EKKO.INCO1]
      ,[EKKO.INCO2]
      ,[EKKO.KTWRT]
      ,[EKKO.KNUMV]
      ,[EKKO.KALSM]
      ,[EKKO.STAFO]
      ,[EKKO.UPINC]
      ,[EKKO.LANDS]
      ,[EKKO.STCEG_L]
      ,[EKKO.STCEG]
      ,[EKKO.ABSGR]
      ,[EKKO.PROCSTAT]
      ,[EKKO.RLWRT]
      ,[EKKO.REVNO]
      ,[EKKO.RETPC]
      ,[EKKO.DPPCT]
      ,[EKKO.DPAMT]
      ,[EKKO.DPDAT]
      ,[EKKO.ZZZL_CURRENCY]
      ,[EKKO.ZZZL_MLPCURR]*/

      ,[EKPO.MENGE] as 'PO Qty'
      ,[EKPO.MEINS] as 'PO UoM'
      ,[EKPO.BPRME] as 'Order Price Unit'
      ,[EKPO.NETWR] as 'Net Order value in PO Curr'
      ,[EKPO.BRTWR] as 'Gross order val in PO Curr'
      ,[EKPO.TXZ01] as 'Short Text'
      ,[EKPO.NETPR] as 'Net Price in P Doc'
      ,[EKPO.PEINH] as 'Price Unit'
      ,[EKPO.MATNR] as 'Material number'
      ,[EKPO.BEDAT]
  FROM [FinanceDa].[dbo].[EKBE_EKPO_EKKO_Benzene2099002]
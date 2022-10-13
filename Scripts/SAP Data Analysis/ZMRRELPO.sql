/* This query is to replicate Transaction code	ZMRRELPO */
Select --top 1000

KO.EBELN, --Purchasing Document Number
PO.EBELP, --Item Number of Purchasing Document
--KO.BUKRS, --Company Code
KO.BSART, --Purchasing Document Type
--PO.LOEKZ, --Deletion indicator in purchasing document
KO.LIFNR, --Vendors account number
VEND.NAME1, --Vendor Name
--KO.SPRAS, --Language Key
KO.EKORG, --Purchasing Organization
KO.EKGRP, --Purchasing Group
KO.BEDAT, --Purchasing Document Date
--KO.KDATB, --Start of Validity Period
--KO.KDATE, --End of Validity Period
--KO.KUNNR, --Customer Number
--KO.KONNR, --Number of principal purchase agreement
KO.BSTYP, --Purchasing Document Category
--PO.MATNR, --Material Number
PO.WERKS, --Plant
PO.MATKL, --Material Group
PO.NETWR, --Net Order Value in PO Currency
PO.PSTYP, --Item category in purchasing document
--PO.KNTTP, --Account Assignment Category
--PO.ABDAT, --Reconciliation date for agreed cumulative quantity
PO.MTART, --Material Type
PO.ELIKZ, --Delivery Completed Indicator
KO.ERNAM, --Name of Person Who Created the Object
PO.MENGE, --Purchase Order Quantity
PO.NETPR, --Net Price in Purchasing Document (in Document Currency)
PO.MEINS, --Purchase Order Unit of Measure
KO.WAERS, --Currency Key
PO.EMATN --Material number

from [Source].[SapEcc].[vtEkpo] as PO  inner join       --Purchasing Document Item 
	 [Source].[SapEcc].[vtEkko] as KO                   --Purchasing Document Header
	 on ko.ebeln = po.ebeln
	 
	 inner join [Source].[SapEcc].[vtlfa1] as VEND      --Vendor table
	 on vend.lifnr = ko.lifnr

	 Where po.werks = 'PKTN' AND             --Plant (PKTN = LCC - Pak Tank - Deer Park)
	       ko.BSART = 'ZB'   AND             --Doc Type (ZB = Raw Material PO)
		   po.MATNR = '000000000002099002'   --Material Benzene

  order by ABDAT asc
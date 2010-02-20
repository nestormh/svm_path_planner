(clear)

(deftemplate Dist
    (slot t1    (type STRING))
    (slot t2    (type STRING))
    (slot miles (type INTEGER)))
	
	(deftemplate triple
    (slot predicate    (type STRING))
    (slot subject    (type STRING))
    (slot object (type STRING)))

(deftemplate Go
    (slot from (type STRING))
    (slot dest (type STRING))
    (slot at   (type STRING))
    (slot miles (type INTEGER)(default 0))
    (multislot route)
)


	
	(reset)
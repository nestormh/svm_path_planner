

































(defrule rule-1
















































































(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?x)
(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?a)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?a)
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?a)
    (object   ?b)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?b)
    (object   ?y)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?b)
    (object   ?d)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
(test (>  (- ?d ?z) 0))
(test (<=  (- ?d ?z) 300))
=>
(assert





(triple
(predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
(subject    ?x)
(object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
)



)
)















(defrule rule-2









































































































































(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?d)
    (object   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
;?d)
(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?f)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?f)
(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?x)
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
    (subject    ?d)
    (object   ?e)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?f)
    (object   ?h)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?f)
    (object   ?g)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (subject    ?d)
    (object   ?y)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?y)
    (object   ?a)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?g)
    (object   ?i)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?e)
    (object   ?j)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?g)
    (object   ?e)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
(test (<=  (- ?j ?i) 50))
(test (> ?h 0))
(test (<=  (- ?a ?z) 50))
=>
(assert





(triple
(predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
(subject    ?x)
(object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
)



)
)


(defrule rule-3





















(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
;?x)
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
    (subject    ?x)
    (object   ?z)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
    (subject    ?x)
    (object   ?y)
  )
=>
(assert





(triple
(predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
(subject    ?x)
(object   http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
)



)
)


(defrule rule-4





























































































































































(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?x)
(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?o)
    (object   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
;?o)
(triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?f)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
;?f)
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
    (subject    ?x)
    (object   ?d)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
    (subject    ?o)
    (object   ?m)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (subject    ?o)
    (object   ?e)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?f)
    (object   ?g)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?g)
    (object   ?e)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
    (subject    ?d)
    (object   ?h)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
    (subject    ?j)
    (object   ?m)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
    (subject    ?j)
    (object   ?k)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
    (subject    ?d)
    (object   ?j)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
    (subject    ?h)
    (object   ?y)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
    (subject    ?h)
    (object   ?i)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?y)
    (object   ?a)
  )
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
(test (=  (- ?k ?i) 1 ))
(test (<=  (- ?a ?z) 50))
=>
(assert





(triple
(predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
(subject    ?x)
(object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
)



)
)










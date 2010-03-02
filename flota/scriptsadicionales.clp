(defrule zlimpiarPosiciones
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarPosicionesAntiguas)
    )
               
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?posicion)
 )
 
  
=> ; (printout t "Activada limpiar:" crlf)
  (retract ?posicionvieja)
;  (retract ?fact)
)

(defrule zlimpiarVelocidades
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarVelocidadesAntiguas)
    )
               
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?c)
    (object   ?posicion)
 )
 
  
=> ; (printout t "Activada limpiar:" crlf)
  (retract ?posicionvieja)
;  (retract ?fact)
)

(defrule zlimpiarTramos
 
   ?fact2 <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarTramosAntiguos)
    )
               
  ?posicionvieja2 <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?posicion)
 )
 
  
=>  
;(printout t "Activada limpiar tramos:" crlf)
  (retract ?posicionvieja2)
;  (retract ?fact)
)


(defrule limpiarAssert
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarPosicionesAntiguas)
    )
               
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?posicion)
     )
 )
 

 
=>;(printout t "Activada limpiar  assert:" crlf)
  (retract ?fact)
)

(defrule limpiarAssertTramos
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarTramosAntiguos)
    )
               
 
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?d)
    (object   ?p)
     )
 )
 
=>;(printout t "Activada limpiar  assert:" crlf)
  (retract ?fact)
)


(defrule zlimpiarEstado
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarEstado)
    )
               
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   ?y)
  )
 
  
=> ; (printout t "Activada limpiar:" crlf)
  (retract ?posicionvieja)
)


(defrule limpiarAssertEstado
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarEstado)
    )
               
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   ?y)
  )
 )
 
 
=>;(printout t "Activada limpiar  assert estado:" crlf)
  (retract ?fact)
)

(defrule limpiarAssertVelocidades
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarVelocidadesAntiguas)
    )
               
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?x)
    (object   ?y)
  )
 )
 
 
=>;(printout t "Activada limpiar  assert estado:" crlf)
  (retract ?fact)
)



(defrule zlimpiarRuta
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarRutas)
    )
               
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
    (subject    ?x)
    (object   ?y)
  )
 
  
=> ;(printout t "Activada limpiar ruta:" ?x crlf)
  (retract ?posicionvieja)
)


(defrule limpiarAssertRutas
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarRutas)
    )
               
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
    (subject    ?x)
    (object   ?y)
  )
 )
 
 
=>;(printout t "Activada limpiar  rutas:" crlf)
  (retract ?fact)
)

;------------------------------------

(defrule zlimpiarDistanciasConflictos
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarDistanciasConflictos)
    )
               
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
    (subject    ?x)
    (object   ?y)
  )
 
  
=> ;(printout t "Activada limpiar conflicto:" ?x crlf)
  (retract ?posicionvieja)
)


(defrule limpiarAssertDistanciasConflictos
 
   ?fact <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarDistanciasConflictos)
    )
               
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
    (subject    ?x)
    (object   ?y)
  )
 )
 
 
=>;(printout t "Activada limpiar  rutas:" crlf)
  (retract ?fact)
)



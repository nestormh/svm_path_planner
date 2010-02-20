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
(printout t "Activada limpiar tramos:" crlf)
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
    (object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
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
    (object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
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
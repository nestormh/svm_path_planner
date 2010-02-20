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

(defrule zlimpiarTramos
 
   ?fact2 <-(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#Accion)
    (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual)
    (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarPosicionesAntiguas)
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
 
 (not (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?posicion)
     )
 )
 
=>;(printout t "Activada limpiar  assert:" crlf)
  (retract ?fact)
)


(defrule conflictos
      
   
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
  
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?x)
    (object   ?v1)
  )  
       
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
                
            
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?y)
    (object   ?a)
  )
                        
                                               
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
    (subject    ?d)
    (object   ?e)
  )
                                            
                                          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (subject    ?d)
    (object   ?y)
  )
                                        
                                      
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?d)
    (object   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
 )
                    
          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
  (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
   
(test (< (- (float ?a) (float ?z)) 50))        
     
(triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?xa)
    (object   ?ca)
  )
  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?xa)
    (object   ?v2)
  )
            
       
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?ca)
    (object   ?e)
  )
                
            
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?e)
    (object   ?aa)
  )
                        
                                               
                                            
                                          
                                        
                                      
                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?ca)
    (object   ?za)
 )
                    
          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?xa)
  (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
   
(test (< (- (float ?aa) (float ?za)) 50))        
 (test (> (float ?v1) 0))
 (test (> (float ?v2) 0))
 
  
  =>  
  (printout t "Disparada la regla Conflictos"  ?x "=" crlf) 
(assert  
        
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
  )
        
        
      
    
) 
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

 
  
(defrule rule-2
      
   
              
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
  (printout t "Disparada la regla 22"  ?x "=" ?v1 crlf) 
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
        
        
          
            
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
    (subject    ?x)
    (object   ?y)
  )
            
            
              
                
                
                  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
    (subject    ?x)
    (object   ?z)
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

  

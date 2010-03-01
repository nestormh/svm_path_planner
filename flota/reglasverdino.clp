
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
(defrule rule-1
      
        
          
            
              
                
                  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
                
                
                  
                    
                      
                        
                          
                            
                              
                                
                                  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?b)
    (object   ?y)
  )
                                
                                
                                  
                                    
                                      
                                        
                                          
 	  (bind ?e (- ?d ?z))
  
                                        
                                        
                                          
                                            
                                              
                                                
                                                  
 	  (test (> ?e 0))  
  
                                                
                                                
                                              
                                            
                                            
                                              
 	  (test (<= ?e 300))  
  
                                            
                                          
                                        
                                      
                                    
                                    
                                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?b)
    (object   ?d)
  )
                                    
                                  
                                
                              
                            
                            
                              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?a)
    (object   ?b)
  )
                            
                          
                        
                        
                          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?a)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
                        
                      
                    
                    
                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
                    
                  
                
              
            
            
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
            
          
        
        
          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
        
      
    
  =>  
(assert  
  
      
        
        
          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#EsperaDistancia)
  )
        
      
    
) 
)

  
  
  
  
  
  
  
  
  
  
  
  
  
  
(defrule rule-2
      
        
          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
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
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?d)
    (object   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
                                    
                                    
                                      
                                        
                                          
                                            
                                              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
    (subject    ?d)
    (object   ?e)
  )
                                            
                                            
                                              
                                                
                                                  
                                                    
                                                      
                                                        
                                                          
                                                            
                                                              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?g)
    (object   ?i)
  )
                                                            
                                                            
                                                              
    
      
        
          
 	  (bind ?k (- ?j ?i))
  
        
        
          
            
              
 	  (test (<= ?k 50))  
  
            
            
              
                
                  
                    
                      
 	  (test (> ?h 0))  
  
                    
                    
                  
                
                
                  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (subject    ?f)
    (object   ?h)
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
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?f)
    (object   ?g)
  )
                                                    
                                                  
                                                
                                                
                                                  
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?f)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
                                                
                                              
                                            
                                          
                                        
                                        
                                          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (subject    ?d)
    (object   ?y)
  )
                                        
                                      
                                    
                                  
                                
                                
                                  
 	  (test (<= ?b 50))  
  
                                
                              
                            
                            
                              
 	  (bind ?b (- ?a ?z))
  
                            
                          
                        
                      
                    
                    
                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
                    
                  
                
              
            
            
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
            
          
        
      
    
  =>  
(assert  
  
      
        
          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#EsperaInterseccionPrioritaria)
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

  
(defrule rule-4
      
        
          
            
              
                
                  
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
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?f)
    (object   ?g)
  )
                                        
                                        
                                          
                                            
                                              
                                                
                                                  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
    (subject    ?x)
    (object   ?d)
  )
                                                
                                                
                                                  
                                                    
                                                      
                                                        
                                                          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
    (subject    ?h)
    (object   ?i)
  )
                                                        
                                                        
                                                          
                                                            
                                                              
    
      
        
          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
    (subject    ?j)
    (object   ?k)
  )
        
        
          
            
              
                
                  
                    
                      
 	  (test (= ?n 1 ))  
  
                    
                    
                      
                        
                          
                            
                              
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
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?o)
    (object   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
                        
                      
                    
                  
                
                
                  
 	  (bind ?n (- ?k ?i))
  
                
              
            
            
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
    (subject    ?j)
    (object   ?m)
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
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
    (subject    ?d)
    (object   ?h)
  )
                                                    
                                                  
                                                
                                              
                                            
                                            
                                              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?g)
    (object   ?e)
  )
                                            
                                          
                                        
                                      
                                    
                                    
                                      
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?f)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
                                    
                                  
                                
                                
                                  
 	  (test (<= ?b 50))  
  
                                
                              
                            
                            
                              
 	  (bind ?b (- ?a ?z))
  
                            
                          
                        
                      
                    
                    
                      
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?z)
  )
                    
                  
                
              
            
            
              
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?x)
    (object   ?c)
  )
            
          
        
        
          
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
        
      
    
  =>  
(assert  
  
      
        
          
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#EsperaOposicion)
  )
        
        
      
    
) 
)

  
  
  
  
  
  
  
  
  
  

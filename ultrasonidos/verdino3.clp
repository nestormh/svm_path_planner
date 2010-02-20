
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
(defrule rule-1
      
        
          
            
              
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo")
    (subject    ?x)
    (object   ?c)
  )
            
            
              
                
                  
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#estaEnTramo")
    (subject    ?c)
    (object   ?y)
  )
                
                
                  
                    
                      
                        
                          
                            
                              
                            
                            
                              
                                
                                  
                                
                                
                                  
                                    
                                      
  (triple
    (predicate "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    (subject    ?d)
    (object   "http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria")
  )
                                    
                                    
                                      
                                        
                                          
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario")
    (subject    ?d)
    (object   ?y)
  )
                                        
                                        
                                          
                                            
                                              
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario")
    (subject    ?d)
    (object   ?e)
  )
                                            
                                            
                                              
                                                
                                                  
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad")
    (subject    "?x")
    (object   "?l")
  )
                                                
                                                
                                                  
                                                    
                                                      
                                                    
                                                    
                                                      
                                                        
                                                          
                                                            
                                                              
    
      
        
          
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud")
    (subject    "?g")
    (object   "?i")
  )
        
        
          
            
              
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneLongitud")
    (subject    "?e")
    (object   "?j")
  )
            
            
              
                
                  
                    
                      
                    
                    
                      
                        
                          
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad")
    (subject    "?f")
    (object   "?h")
  )
                        
                        
                          
                            
                            
                              
                            
                          
                        
                      
                    
                  
                
                
                  
                
              
            
          
        
      
    
    
      
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#estaEnTramo")
    (subject    ?g)
    (object   ?e)
  )
                                                  
                                                            
                                                            
                                                              
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo")
    (subject    ?f)
    (object   ?g)
  )
                                                            
                                                          
                                                        
                                                        
                                                          
  (triple
    (predicate "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    (subject    ?f)
    (object   "http://www.isaatc.ull.es/Verdino.owl#Vehiculo")
  )
                                                        
                                                      
                                                    
                                                  
                                                
                                              
                                            
                                          
                                        
                                      
                                    
                                  
                                
                              
                            
                          
                        
                        
                          
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneLongitud")
    (subject    "?y")
    (object   "?a")
  )
                        
                      
                    
                    
                      
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud")
    (subject    "?c")
    (object   "?z")
  )
                    
                  
                
              
            
          
        
        
          
  (triple
    (predicate "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    (subject    ?x)
    (object   "http://www.isaatc.ull.es/Verdino.owl#Vehiculo")
  )
        
      
    
  =>  
(assert  
  
      
        
        
          
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tieneEstado")
    (subject    ?x)
    (object   ?EnEspera)
  )
        
      
    
) 
)

  
  
(defrule rule-2
      
        
          
  (triple
    (predicate "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    (subject    ?x)
    (object   "http://www.isaatc.ull.es/Verdino.owl#Tramo")
  )
        
        
          
            
              
                
                
                  
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial")
    (subject    ?x)
    (object   ?z)
  )
                
              
            
            
              
  (triple
    (predicate "http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal")
    (subject    ?x)
    (object   ?y)
  )
            
          
        
      
    
  =>  
(assert  
  
      
        
          
  (triple
    (predicate "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    (subject    ?x)
    (object   "http://www.isaatc.ull.es/Verdino.owl#TramoPintable")
  )
        
        
      
    
) 
)

  
  
  
  

(import java.awt.*)
(import jess.awt.*)

(defglobal ?*frame* = (new java.awt.Frame "Verdino Simulator"))



(deffunction closer (?event)
   (if (= (call ?event getID) (get-member ?event WINDOW_CLOSING)) then 
     (call (get ?event source) dispose)
     ))

(deffunction painter (?canvas ?graph)
  (bind ?x (get-member (call ?canvas getSize) width))
  (bind ?y (get-member (call ?canvas getSize) height))
  (?graph setColor (get-member java.awt.Color blue))
  (?graph fillRect 0 0 ?x ?y)
  ;(?graph setColor (get-member java.awt.Color red))
  ;(?graph drawLine 0 0 ?x ?y)
  ;(?graph drawLine ?x 0 0 ?y)
  )



(?*frame* setSize 700 700)
(?*frame* setVisible TRUE)
(?*frame* addWindowListener (new jess.awt.WindowListener closer (engine)))  
  (bind ?c (new jess.awt.Canvas painter (engine)))

  (?*frame* add "Center" ?c)
  (?c setSize 700 700)
 (bind ?g (?c getGraphics))
 (bind ?x (get-member (call ?c getSize) width))
  
 ;(printout t "     desde " ?jp "," ?x)
  (?*frame* pack)
 ; (?*frame* show)
  

  
  
  
  
(defrule rule-11
      
            
              
 (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
  )
  
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
   (subject   ?x)
   (object    ?pgi)
  )  
            
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
   (subject   ?x)
   (object    ?pgf)
  )               
                


 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   ?pgf)
   (object    ?yf)
  )  
                  
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   ?pgf)
   (object    ?xf)
  )  
        
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   ?pgi)
   (object    ?yi)
  )  
                  
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   ?pgi)
   (object    ?xi)
  )        
	  
    
  =>  
(printout t "Disparada la regla de pintar sobre Tramo " ?x) 
(printout t "     desde " ?xi "," ?yi)
(printout t "     hasta " ?xf "," ?yf crlf)
(?g setColor (get-member java.awt.Color green))
;(?g fillRect 100 200 300 (integer "400))
(?g drawLine (integer ?xi) (integer ?yi) (integer ?xf) (integer ?yf))
(?g setColor (get-member java.awt.Color red))
(bind ?deltaX (- (integer ?xf)  (integer ?xi)));
(bind ?deltaY (- (integer ?yf)  (integer ?yi)));
(bind ?frac 0.1);
(bind ?frac2 0.01);
	;g.drawLine(x0,y0,x1,y1);
(?g drawLine 
	(+ (integer ?xi)  (integer ( + (* (- 1 ?frac) ?deltaX) (* ?frac2 ?deltaY))))
	        (+ (integer ?yi)  (integer ( - (* (- 1 ?frac) ?deltaY) (* ?frac2 ?deltaX))))
			(integer ?xf)  (integer ?yf));
(?g drawLine 
	(+ (integer ?xi)  (integer ( - (* (- 1 ?frac) ?deltaX) (* ?frac2 ?deltaY))))
	        (+ (integer ?yi)  (integer ( + (* (- 1 ?frac) ?deltaY) (* ?frac2 ?deltaX))))
			(integer ?xf)  (integer ?yf));

			


			;g.drawLine(x0 + (int)((1-frac)*deltaX - frac*deltaY),
	;	   y0 + (int)((1-frac)*deltaY + frac*deltaX),
	;	   x1, y1);


)


(defrule rule-22
 (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (subject    ?x)
    (object   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
  )
   ?antiguavelocidad<-
   (triple(predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   ?x)
   (object    ?velocidad)
  )
=>
(printout t "Disparada la regla de Proximo a Cruce " ?x crlf) 
(assert
 (triple
 (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   ?x)
   (object    0.0)
  )
  )
(retract ?antiguavelocidad)
)

(defrule mvtoverdinos
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   ?verdino)
   (object    ?velocidad)
  )
  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?verdino)
    (object   ?c)
  )
            
       
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
                
            
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?y)
    (object   ?longitudtramo)
  )
                        
                                               
              
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?posicion)
 )
                    
  (test (< (+ ?posicion ?velocidad) ?longitudtramo))
  (test (> ?velocidad 0))  
 
=>
(bind ?nuevaposicion (+ ?posicion ?velocidad))
(retract ?posicionvieja)
(printout t "Posicion de  " ?verdino ": " ?nuevaposicion crlf) 
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (subject   ?c)
   (object    ?nuevaposicion)
  )
)  

)

(defrule cambioTramos
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   ?verdino)
   (object    ?velocidad)
  )
  
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (subject    ?verdino)
    (object   ?c)
  )
            
       
  ?antiguotramo <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (subject    ?c)
    (object   ?y)
  )
                
            
  (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (subject    ?y)
    (object   ?longitudtramo)
  )
                        
                                               
              
  ?posicionvieja <- (triple
    (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (subject    ?c)
    (object   ?posicion)
 )
          
 (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
   (subject   ?verdino)
   (object    ?ruta)
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   ?ruta)
   (object    ?tramoorden)
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   ?tramoorden)
   (object    ?orden)
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (subject   ?tramoorden)
   (object    ?y)
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   ?ruta)
   (object    ?tramoordennuevo)
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   ?tramoordennuevo)
   (object    ?nuevoorden&:(eq ?nuevoorden (+ (integer ?orden) 1)))
  )
  
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (subject   ?tramoordennuevo)
  (object    ?ynuevo)
  )
  
          
  (test (>= (+ ?posicion ?velocidad) ?longitudtramo))
  (test (> ?velocidad 0))  
 
=>
(bind ?nuevaposicion 0)
(retract ?posicionvieja)
(retract ?antiguotramo)
(printout t "Posicion de  " ?verdino ": " ?ynuevo " : " ?nuevaposicion crlf) 
;(printout t "Posicion de  " ?verdino ": Desconocida ---" ?orden crlf) 

(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (subject   ?c)
   (object    ?nuevaposicion)
  )
)  

(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
   (subject   ?c)
   (object    ?ynuevo)
  )
)  
; "revivir" a los parados por éste

)


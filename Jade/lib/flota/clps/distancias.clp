

;(reset)

;;
;; Invert all distances
;;
;(defquery Distances
;    ?dist <- (Dist (t1 ?t1)(t2 ?t2)))
;
;(bind ?res (run-query* Distances))
;(while (?res next)
    ;(duplicate (?res get dist) (t1 (?res get t2))(t2 (?res get t1)))
;	)

;; How to go from one city to all immediate neighbors
;;
(defrule Advance
 ?d <- (Dist (t1 ?at)(t2 ?t2)(miles ?miles))
     
   ?g <- (Go   (from ?from)(dest ?dest )(at ?at)(miles ?tofrom)(route $?route))
     =>
    (bind ?msum (+ ?tofrom ?miles))
    (bind ?eroute (create$ $?route ?t2))
    (assert (Go (from ?from)(dest ?dest)(at ?t2)(miles ?msum)(route ?eroute)))
)

;; Eliminate the longer of two routes
;;
(defrule Minimum
    ?g1 <- (Go (from ?from)(dest ?dest )(at ?at)(miles ?m1))
    ?g2 <- (Go (from ?from)(dest ?dest )(at ?at)(miles ?m2 & ~ ?m1))
    =>
    (retract (if (< ?m1 ?m2) then ?g2 else ?g1))
)

(defmodule RESULT)

;; Remove all intermediary results
;;
(defrule Clean
    ?g <- (Go (dest ?dest)(at ?at & ~ ?dest))
    =>
    (retract ?g)
)

;; Display the solution.
;;
(defrule Show
  
    ?g <- ( Go (from ?from) (dest ?dest)(at ?dest)(miles ?miles)(route $?route))
    =>

   ;(printout t ?from " to " ?dest ": " ?miles " miles. " ?route crlf)
   (retract ?g)
  (bind ?resto (implode$ $?route))
  (assert  
    (triple
     (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRutaCalculada)
     (subject    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
     (object   ?resto)
    ))
	(store RUTA ?route)
	(store DISTANCIA ?miles)
)





;; Launcher for computing a minimum distance between two cities.
;;
(deffunction compDist (?from ?dest)
    (assert (Go (from ?from)(dest ?dest)(at ?from)(route ?from)))
    (run)
    (focus RESULT)
    (run)
)

;(compDist Indianapolis Columbus)

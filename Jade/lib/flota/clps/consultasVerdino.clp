(defquery buscaSujetosConObjeto
  "Devuelve los sujetos en triples de Verdino"
  (declare (variables ?predicado ?obj))
  (triple (predicate ?predicado) (subject ?sub) (object ?obj)))
  
(defquery buscaObjetosConSujeto
  "Devuelve los objetos en triples de Verdino"
  (declare (variables ?predicado ?sub))
  (triple (predicate ?predicado) (subject ?sub) (object ?obj)))
  
  (defquery buscaSujetos
  "Devuelve los sujetos en triples de Verdino"
  (declare (variables ?obj))
  (triple (predicate ?predicado) (subject ?sub) (object ?obj)))
  
 
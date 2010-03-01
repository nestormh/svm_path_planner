

  ;;; Declaring the triple template ---------------------------------
(deftemplate triple "Template representing a triple "
 (slot predicate (default "")) 
 (slot subject   (default "")) 
 (slot object    (default ""))
)

;;; ---------------- copy and paste from rdfmt.clp ------------------

;;; ------------------- RDF axiomatic triples -----------------------

;;; ------------------- RDF semantic conditions ---------------------

;;; ------------------- RDFS semantic conditions --------------------

;;; ------------------- RDFS axiomatic triples ----------------------

;;; ----------- end of predefined RDF(S) facts and rules ------------


;;; ---------- OWL model theoretic semantics (RDF-compatible) ----------

;;; Conditions concerning the parts of the OWL universe and syntactic categories

(deffacts OWL_universe

  ;;; This defines IOC owl:Class as the set of OWL classes.
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#Class)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#Class)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
 
 ;;; This defines IDC as the set of OWL datatypes.
 ;;; referring to rdfmt.clp --- rdfs:Datatype rdfs:subClassOf rdfs:Class .
    
 ;;; This defines IOR owl:Restriction as the set of OWL restrictions.
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#Restriction)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#Restriction)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
  
 ;;; This defines IOT owl:Thing as the set of OWL individuals.
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#Thing)
   (object    http://www.w3.org/2002/07/owl#Class)
  )

 ;;; This defines owl:Nothing.
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#Nothing)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
  
  ;;; This defines rdfs:Literal.
  ;;; referring to rdfmt.clp --- rdfs:Literal rdf:type rdfs:Class . 

  ;;; interesting OWL!!! rdfs:Literal rdf:type rdfs:Datatype.
  ;;; referring to rdfmt.clp --- RDFS_semantic_conditions_datatype
 
  ;;; This defines IOOP owl:ObjectProperty as the set of OWL individual-valued properties. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#ObjectProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#ObjectProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  ) 

  ;;; This defines IODP owl:DatatypeProperty as the set of OWL datatype properties. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#DatatypeProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#DatatypeProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  ) 

  ;;; This defines IOAP owl:AnnotationProperty as the set of OWL annotation properties. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#AnnotationProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#AnnotationProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  ) 

  ;;; This defines IOXP owl:OntologyProperty as the set of OWL ontology properties. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#OntologyProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  (triple 
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.w3.org/2002/07/owl#OntologyProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  ) 

  ;;; This defines IX owl:Ontology as the set of OWL ontologies. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#Ontology)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )

  ;;; This defines IAD owl:AllDifferent. 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#AllDifferent)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
       
)


;;; OWL built-in syntactic classes and properties

(deffacts OWL_built_in_syntax

;;; I(owl:FunctionalProperty), I(owl:InverseFunctionalProperty), I(owl:SymmetricProperty),
;;; I(owl:TransitiveProperty), I(owl:DeprecatedClass), and I(owl:DeprecatedProperty) are in CI. 

  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#FunctionalProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#InverseFunctionalProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#SymmetricProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#TransitiveProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#DeprecatedClass)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#DeprecatedProperty)
   (object    http://www.w3.org/2000/01/rdf-schema#Class)
  )

;;; I(owl:equivalentClass), I(owl:disjointWith), I(owl:equivalentProperty), 
;;; I(owl:inverseOf), I(owl:sameAs), I(owl:differentFrom), I(owl:complementOf), 
;;; I(owl:unionOf), I(owl:intersectionOf), I(owl:oneOf), I(owl:allValuesFrom), 
;;; I(owl:onProperty), I(owl:someValuesFrom), I(owl:hasValue), I(owl:minCardinality), 
;;; I(owl:maxCardinality), I(owl:cardinality), and I(owl:distinctMembers) are all in PI. 

  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#equivalentClass)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#disjointWith)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#equivalentProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
   (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#inverseOf)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#sameAs)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#differentFrom)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#complementOf)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#unionOf)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#intersectionOf)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#oneOf)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#allValuesFrom)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#onProperty)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#someValuesFrom)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#hasValue)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#minCardinality)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#maxCardinality)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#cardinality)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#distinctMembers)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Property)
  )

;;; I(owl:versionInfo), I(rdfs:label), I(rdfs:comment), I(rdfs:seeAlso),  I(rdfs:isDefinedBy) in IOAP. 

  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#versionInfo)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2000/01/rdf-schema#label)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2000/01/rdf-schema#comment)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2000/01/rdf-schema#seeAlso)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2000/01/rdf-schema#isDefinedBy)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  )
  
;;; I(owl:imports), I(owl:priorVersion), I(owl:backwardCompatibleWith), I(owl:incompatibleWith) in IOXP. 
 
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#imports)
   (object    http://www.w3.org/2002/07/owl#OntologyProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#priorVersion)
   (object    http://www.w3.org/2002/07/owl#OntologyProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#backwardCompatibleWith)
   (object    http://www.w3.org/2002/07/owl#OntologyProperty)
  )
  
  (triple 
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.w3.org/2002/07/owl#incompatibleWith)
   (object    http://www.w3.org/2002/07/owl#OntologyProperty)
  )

  ;;; This defines IL as the set of OWL lists.
  ;;; referring to rdfmt.clp --- rdf:List rdf:type rdfs:Class .
  
  ;;; This defines rdf:nil.
  ;;; referring to rdfmt.clp --- rdf:nil rdf:type rdf:List .
   
)

;;; Characteristics of OWL classes, datatypes, and properties

;;; Instances of OWL classes are OWL individuals.
(defrule OWL_characteristics_Class
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#Class)
  ) 
  =>
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf) 
                  (subject   ?e)
                  (object    http://www.w3.org/2002/07/owl#Thing)
           ) 
   ) 
)

;;; OWL datatype are special kinds of rdfs:Literal.
;;; referring to rdfmt.clp --- defrule RDFS_semantic_conditions_datatype

;;; OWL dataranges are special kinds of datatypes.
(defrule OWL_characteristics_DataRange
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#DataRange)
  ) 
   =>
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf) 
                  (subject   ?e)
                  (object    http://www.w3.org/2000/01/rdf-schema#Literal)
           )
  )
)

;;; Values for individual-valued properties are OWL individuals.
(defrule OWL_characteristics_ObjectProperty
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  ) 
  (triple (predicate ?e) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?x)
                  (object   http://www.w3.org/2002/07/owl#Thing)
           ) 
   ) 
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Thing)
           ) 
   )    
)

;;; Values for datatype properties are literal values.
(defrule OWL_characteristics_DatatypeProperty
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  ) 
  (triple (predicate ?e) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Thing)
           ) 
   ) 
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?y)
                  (object    http://www.w3.org/2000/01/rdf-schema#Literal)
           ) 
   )    
)

;;; Values for annotation properties are less unconstrained.
;;; two rules dealing with (?y rdf:type owl:Thing | rdfs:Literal)
(defrule OWL_characteristics_AnnotationProperty_subject
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  ) 
  (triple (predicate ?e) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Thing)
           ) 
   )   
)
;;; By default, values-range for annotation properties are owl:Thing
(defrule OWL_characteristics_AnnotationProperty_object
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#AnnotationProperty)
  ) 
  (triple (predicate ?e) (subject ?x) (object ?y))
  (not (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
               (subject   ?y)
               (object    http://www.w3.org/2000/01/rdf-schema#Literal)
       ) 
   )   
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Thing)
           ) 
   )   
)

;;; Ontology properties relate ontologies to other ontologies.
(defrule OWL_characteristics_OntologyProperty
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?e)
   (object    http://www.w3.org/2002/07/owl#OntologyProperty)
  ) 
  (triple (predicate ?e) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?x)
                  (object   http://www.w3.org/2002/07/owl#Ontology)
           ) 
   ) 
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) 
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Ontology)
           ) 
   )    
)

;;; interesting OWL!!! thinking about OWA!!!
;;; interesting OWL!!! By default, both individual-valued and datatype properties are functional properties!
;;; interesting OWL!!! By default, individual-valued are inverse functional,symmetric,transitive properties!


;;; Both individual-valued and datatype properties can be functional properties.
;;; --- if ---
(defrule OWL_characteristics_FunctionalProperty_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty|http://www.w3.org/2002/07/owl#DatatypeProperty)
  )  
  (triple (predicate ?c) (subject ?x) (object ?y1))
  (triple (predicate ?c) (subject ?x) (object ?y2))
  =>
  (assert (triple (predicate http://www.w3.org/2002/07/owl#sameAs) 
                  (subject   ?y1)
                  (object    ?y2)
           ) 
   )    
)
;;; --- only if ---
(defrule OWL_characteristics_FunctionalProperty_only_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty|http://www.w3.org/2002/07/owl#DatatypeProperty)
  ) 
  (not
    (and 
      (triple (predicate ?c) (subject ?x) (object ?y1))
      (triple (predicate ?c) (subject ?x) (object ?y2))
      (not (triple (predicate http://www.w3.org/2002/07/owl#sameAs) (subject ?y1) (object ?y2)))
    )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?c)
                  (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
          )
  )    
)

;;; Only individual-valued properties can be inverse functional properties.
;;; --- if ---
(defrule OWL_characteristics_InverseFunctionalProperty_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#InverseFunctionalProperty)
  )
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )  
  (triple (predicate ?c) (subject ?x1) (object ?y))
  (triple (predicate ?c) (subject ?x2) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/2002/07/owl#sameAs) 
                  (subject   ?x1)
                  (object    ?x2)
           ) 
   )    
)
;;; --- only if ---
(defrule OWL_characteristics_InverseFunctionalProperty_only_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  ) 
  (not
    (and 
      (triple (predicate ?c) (subject ?x1) (object ?y))
      (triple (predicate ?c) (subject ?x2) (object ?y))
      (not (triple (predicate http://www.w3.org/2002/07/owl#sameAs) (subject ?x1) (object ?x2)))
    )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?c)
                  (object    http://www.w3.org/2002/07/owl#InverseFunctionalProperty)
          )
  )    
)

;;; Only individual-valued properties can be symmetric properties.
;;; --- if ---
(defrule OWL_characteristics_SymmetricProperty_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#SymmetricProperty)
  )
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )  
  (triple (predicate ?c) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate ?c) (subject ?y) (object ?x))) 
)
;;; --- only if ---
(defrule OWL_characteristics_SymmetricProperty_only_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  ) 
  (not
    (and 
      (triple (predicate ?c) (subject ?x) (object ?y))
      (not (triple (predicate ?c) (subject ?y) (object ?x)))
    )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?c)
                  (object    http://www.w3.org/2002/07/owl#SxmmetricProperty)
          )
  )    
)

;;; Only individual-valued properties can be transitive properties.
;;; --- if ---
(defrule OWL_characteristics_TransitiveProperty_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#TransitiveProperty)
  )
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )  
  (triple (predicate ?c) (subject ?x) (object ?y))
  (triple (predicate ?c) (subject ?y) (object ?z))
  =>
  (assert (triple (predicate ?c) (subject ?x) (object ?z))) 
)
;;; --- only if ---
(defrule OWL_characteristics_TransitiveProperty_only_if
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   ?c)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  ) 
  (not
    (and 
      (triple (predicate ?c) (subject ?x) (object ?y))
      (triple (predicate ?c) (subject ?y) (object ?z))
      (not (triple (predicate ?c) (subject ?x) (object ?z)))
    )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?c)
                  (object    http://www.w3.org/2002/07/owl#TransitiveProperty)
          )
  )    
)

;;; If-and-only-if conditions for rdfs:subClassOf, rdfs:subPropertyOf, rdfs:domain, and rdfs:range

;;; --- if ---
;;; referring to rdfmt.clp --- RDFS_semantic_conditions_subClassOf,subPropertyOf,domain,range

;;; --- only if ---
;;; interesting OWL!!! empty set is subclass of all others!
;;; too strong semantics to be impossible!!!


;;; Characteristics of OWL vocabulary related to equivalence

;;; --- if ---

(defrule OWL_characteristics_equivalentClass_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#equivalentClass) (subject ?x) (object ?y))
  =>
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?x)
                    (object    http://www.w3.org/2002/07/owl#Class)
            ) 
  )
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?y)
                    (object    http://www.w3.org/2002/07/owl#Class)
            ) 
  )
)

(defrule OWL_characteristics_equivalentClass_relationship
  (triple (predicate http://www.w3.org/2002/07/owl#equivalentClass) (subject ?x) (object ?y))
  (test (neq 0 (str-compare ?x ?y)))  
  =>
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf) (subject ?x) (object ?y))) 
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf) (subject ?y) (object ?x)))    
)

(defrule OWL_characteristics_disjointWith_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#disjointWith) (subject ?x) (object ?y))
  =>
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?x)
                    (object    http://www.w3.org/2002/07/owl#Class)
            ) 
  )
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?y)
                    (object    http://www.w3.org/2002/07/owl#Class)
            ) 
  )
)

(defrule OWL_characteristics_disjointWith_relationship
  (triple (predicate http://www.w3.org/2002/07/owl#disjointWith) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?o1) (object ?x))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?o2) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#differentFrom) (subject ?o1) (object ?o2))) 
)

;;; By default, values-domain-range for equivalentProperty are DatatypeProperty
(defrule OWL_characteristics_equivalentProperty_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#equivalentProperty) (subject ?x) (object ?y))
  (not (and (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?x)
                    (object    http://www.w3.org/2002/07/owl#ObjectProperty)
            )
            (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?y)
                    (object    http://www.w3.org/2002/07/owl#ObjectProperty)
            )
       )
  )
  =>
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?x)
                    (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
            ) 
  )
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?y)
                    (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
            ) 
  )
)

(defrule OWL_characteristics_equivalentProperty_relationship
  (triple (predicate http://www.w3.org/2002/07/owl#equivalentProperty) (subject ?x) (object ?y))
  (test (neq 0 (str-compare ?x ?y)))  
  =>
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subPropertyOf) (subject ?x) (object ?y))) 
  (assert (triple (predicate http://www.w3.org/2000/01/rdf-schema#subPropertyOf) (subject ?y) (object ?x)))    
)

(defrule OWL_characteristics_inverseOf_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#inverseOf) (subject ?x) (object ?y))
  =>
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?x)
                    (object    http://www.w3.org/2002/07/owl#ObjectProperty)
            ) 
  )
  (assert   (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                    (subject   ?y)
                    (object    http://www.w3.org/2002/07/owl#ObjectProperty)
            ) 
  )
)

(defrule OWL_characteristics_inverseOf_relationship
  (triple (predicate http://www.w3.org/2002/07/owl#inverseOf) (subject ?x) (object ?y))
  (triple (predicate ?x) (subject ?u) (object ?v))  
  =>
  (assert (triple (predicate ?y) (subject ?v) (object ?u))) 
)

(defrule OWL_characteristics_sameAs
  (triple (predicate http://www.w3.org/2002/07/owl#sameAs) (subject ?x) (object ?y))
  (test (neq 0 (str-compare ?x ?y)))  
  =>
  ;(printout t Error---sameAs!!!  ?x  is not same as  ?y crlf) 
)

(defrule OWL_characteristics_differentFrom
  (triple (predicate http://www.w3.org/2002/07/owl#differrentFrom) (subject ?x) (object ?y))
  (test (eq 0 (str-compare ?x ?y)))  
  =>
  (printout t Error---differentFrom!!!  ?x  is not different from  ?y crlf) 
)

;;; --- only if ---
;;; intereting OWL!!! ignore all the only-if strong semantics!

;;; Conditions on OWL vocabulary related to boolean combinations and sets

;;; ignore the rdfs:List

;;; --- owl:complementOf ---

(defrule OWL_complementOf_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#complementOf) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )  
)

;;; considering---there is no classical negation in Jess, hence only error messages!
(defrule OWL_complementOf_if
  (triple (predicate http://www.w3.org/2002/07/owl#complementOf) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/2002/07/owl#disjointWith) (subject ?x) (object ?y)))
)

(defrule OWL_complementOf_only_if
  (triple (predicate http://www.w3.org/2002/07/owl#complementOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
          (subject   ?u)
          (object    http://www.w3.org/2002/07/owl#Thing)
  )
  (not (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?y)))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
)

;;; --- owl:unionOf ---

(defrule OWL_unionOf_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#unionOf) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )  
)

;;; considering---there is no instance of unionOf, hence assign one arbitrarily!
(defrule OWL_unionOf_subset
  (triple (predicate http://www.w3.org/2002/07/owl#unionOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (not (and (triple (predicate http://www.w3.org/2002/07/owl#unionOf) (subject ?x) (object ?v))
	    (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?v))
       )
  )
  =>
  (printout t Attention---unionOf  ?x  !!! arbitrarily assign  ?u  belonging to  ?y crlf)
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?y)))
  (halt)
)

(defrule OWL_unionOf_supset
  (triple (predicate http://www.w3.org/2002/07/owl#unionOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
  (halt)
)


;;; --- owl:intersectionOf ---

(defrule OWL_intersectionOf_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#intersectionOf) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2002/07/owl#Class)
          )
  )  
)

(defrule OWL_intersectionOf_subset
  (triple (predicate http://www.w3.org/2002/07/owl#intersectionOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?y)))
)

(defrule OWL_intersectionOf_supset
  (triple (predicate http://www.w3.org/2002/07/owl#intersectionOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?y))
  (not (and (triple (predicate http://www.w3.org/2002/07/owl#intersectionOf) (subject ?x) (object ?v))
            (not (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?v)))
       )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
  (halt)
)

;;; --- owl:oneOf ---

(defrule OWL_oneOf_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#oneOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?y) (object ?s))
  =>
  (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#Thing))
   then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                        (subject   ?x)
                        (object    http://www.w3.org/2002/07/owl#Class)
                )
        )
   else (if (eq 0 (str-compare ?s http://www.w3.org/2000/01/rdf-schema#Literal))
         then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                              (subject   ?x)
                              (object    http://www.w3.org/2000/01/rdf-schema#Datatype)
                      )
              )
         eles (printout t Error---oneOf! ?y should be OWL Thing/RDFS Literal! crlf) 
         )
   )
)


;;; considering---there is no instance of oneOf, hence assign one arbitrarily!
(defrule OWL_oneOf_subset
  (triple (predicate http://www.w3.org/2002/07/owl#oneOf) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (not (and (triple (predicate http://www.w3.org/2002/07/owl#oneOf) (subject ?x) (object ?v))
	    (or (test (eq 0 (str-compare ?u ?v)))
	        (not (triple (predicate http://www.w3.org/2002/07/owl#sameAs) (subject ?u) (object ?v))))
       )
  )
  =>
  (printout t Attention---oneOf  ?x  !!! arbitrarily assign  ?u  same as  ?y crlf)
  (assert (triple (predicate http://www.w3.org/2002/07/owl#sameAs) (subject ?u) (object ?y))) 
  (halt)
)

(defrule OWL_oneOf_supset
  (triple (predicate http://www.w3.org/2002/07/owl#oneOf) (subject ?x) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?y) (object ?x)))
)

;;; Conditions on OWL restrictions

;;; --- owl:allValuesFrom ---

(defrule OWL_allValuesFrom_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#allValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?p) (object ?s))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#ObjectProperty))
   then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                        (subject   ?y)
                        (object    http://www.w3.org/2002/07/owl#Class)
                )
        )
   else (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#DatatypeProperty))
         then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                              (subject   ?y)
                              (object    http://www.w3.org/2000/01/rdf-schema#Datatype)
                      )
              )
         eles (printout t Error---allValuesFrom! ?p should be Object/Datatype Property! crlf) 
         )
   )
)

(defrule OWL_allValuesFrom_subset
  (triple (predicate http://www.w3.org/2002/07/owl#allValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (triple (predicate ?p) (subject ?u) (object ?v))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?v) (object ?y)))
)

(defrule OWL_allValuesFrom_supset
  (triple (predicate http://www.w3.org/2002/07/owl#allValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate ?p) (subject ?u) (object ?v))
  (not (and (triple (predicate ?p) (subject ?o) (object ?v))
            (not (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?v) (object ?y)))
       )
  )
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
)

;;; --- owl:someValuesFrom ---

(defrule OWL_someValuesFrom_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#someValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?p) (object ?s))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#ObjectProperty))
   then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                        (subject   ?y)
                        (object    http://www.w3.org/2002/07/owl#Class)
                )
        )
   else (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#DatatypeProperty))
         then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                              (subject   ?y)
                              (object    http://www.w3.org/2000/01/rdf-schema#Datatype)
                      )
              )
         ;else (printout t Error---someValuesFrom! ?p should be Object/Datatype Property! crlf) 
         )
   )
)

;;; considering---there is no instance of someValuesFrom, hence assign one arbitrarily!
(defrule OWL_someValuesFrom_subset
  (triple (predicate http://www.w3.org/2002/07/owl#someValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (triple (predicate ?p) (subject ?u) (object ?v))
  (not (and (triple (predicate ?p) (subject ?u) (object ?o))
            (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?o) (object ?s))
            (test (eq 0 (str-compare ?s ?y)))
       )
  )
  =>
  (printout t Attention---someValuesFrom  ?x  !!! arbitrarily assign  ?v  belonging to  ?y crlf)
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?v) (object ?y)))
  (halt)
)

(defrule OWL_someValuesFrom_subset_append
  (triple (predicate http://www.w3.org/2002/07/owl#someValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (not (triple (predicate ?p) (subject ?u) (object ?v)))
  =>
  ;;; caution
  ;(printout t Caution! --- owl:someValuesFrom  ?x  *sub*  crlf there is nothing related to  ?p crlf)
)

(defrule OWL_someValuesFrom_supset
  (triple (predicate http://www.w3.org/2002/07/owl#someValuesFrom) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate ?p) (subject ?u) (object ?v))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?v) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
  (halt)
)

;;; --- owl:hasValue ---

(defrule OWL_hasValue_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#hasValue) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?p) (object ?s))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#ObjectProperty))
   then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                        (subject   ?y)
                        (object    http://www.w3.org/2002/07/owl#Thing)
                )
        )
   else (if (eq 0 (str-compare ?s http://www.w3.org/2002/07/owl#DatatypeProperty))
         then (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                              (subject   ?y)
                              (object    http://www.w3.org/2000/01/rdf-schema#Literal)
                      )
              )
         eles (printout t Error---allValuesFrom! ?p should be Object/Datatype Property! crlf) 
         )
   )
)

(defrule OWL_hasValue_subset
  (triple (predicate http://www.w3.org/2002/07/owl#hasValue) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  =>
  (assert (triple (predicate ?p) (subject ?u) (object ?y)))
)

(defrule OWL_hasValue_supset
  (triple (predicate http://www.w3.org/2002/07/owl#hasValue) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate ?p) (subject ?u) (object ?y))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
)

;;; define a query to count the cardinalty

(defquery OWL_cardinality_query 
    (declare (variables ?P ?S))
    (triple (predicate ?P) (subject ?S) (object ?O) )
)

;;; --- owl:minCardinality ---

(defrule OWL_minCardinality_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#minCardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2000/01/rdf-schema#Literal)
          )
  )  
  (if (> 0 ?y) then (printout t Error---minCardinality!!!   ?x  the number of  ?y  is negative!  crlf))

)

(defrule OWL_minCardinality_subset
  (triple (predicate http://www.w3.org/2002/07/owl#minCardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (test (> ?y (count-query-results OWL_cardinality_query ?p ?u)))  
  =>
 (printout t Error---maxCardinality!!!  ?x  its property  ?p  cardinality is  much less than   ?y  crlf)
)

;;; --- owl:maxCardinality ---

(defrule OWL_maxCardinality_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#maxCardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2000/01/rdf-schema#Literal)
          )
  )  
  (if (> 0 ?y) then (printout t Error---maxCardinality!!!   ?x  the number of  ?y  is negative!  crlf))

)

(defrule OWL_maxCardinality_supset
  (triple (predicate http://www.w3.org/2002/07/owl#maxCardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate ?p) (subject ?u) (object ?v))
  =>
  (if (>= ?y (count-query-results OWL_cardinality_query ?p ?u)) then
    (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
  )
  (halt)
)

;;; --- owl:cardinality ---

(defrule OWL_cardinality_domainrange
  (triple (predicate http://www.w3.org/2002/07/owl#cardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  =>
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?x)
                  (object    http://www.w3.org/2002/07/owl#Restriction)
          )
  )
  (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
                  (subject   ?y)
                  (object    http://www.w3.org/2000/01/rdf-schema#Literal)
          )
  )  
  (if (> 0 ?y) then (printout t Error---cardinality!!!   ?x  the number of  ?y  is negative!  crlf))
)



(defrule OWL_cardinality_supset
  (triple (predicate http://www.w3.org/2002/07/owl#cardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate ?p) (subject ?u) (object ?v))
  =>
  (if (= ?y (count-query-results OWL_cardinality_query ?p ?u)) then
    (assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x)))
  )
  (halt)
)

;;; ------------- end of OWL predefined facts and rules -------------

(reset)

;;; ------ adding Jess assertions transformed from OWL2Jess ---------









; Las reglas que dan problemas de transformación en el fichero de entidad -----------

  (defrule OWL_cardinality_subset
  (triple (predicate http://www.w3.org/2002/07/owl#cardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (test (<> ?y (count-query-results OWL_cardinality_query ?p ?u)))  
  =>
 (printout t Error---minCardinality!!!  ?x  its property  ?p  cardinality is not equal   ?y  crlf)
)



(defrule OWL_maxCardinality_subset
  (triple (predicate http://www.w3.org/2002/07/owl#maxCardinality) (subject ?x) (object ?y))
  (triple (predicate http://www.w3.org/2002/07/owl#onProperty) (subject ?x) (object ?p))
  (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject ?u) (object ?x))
  (test (< ?y (count-query-results OWL_cardinality_query ?p ?u)))  
  =>
 (printout t Error---maxCardinality!!!  ?x  its property  ?p  cardinality is  much more than   ?y  crlf)
)

;---- Reglas transformadas de la ontología
   
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65551)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65551)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65564)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N65564)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N65564)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65564)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65577)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65577)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65577)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65590)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N65590)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65590)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65590)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-4)
   (object    http://www.w3.org/2003/11/swrl#Imp)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_34)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_34)
   (object    50.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_34)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65627)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65627)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65627)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65627)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65627)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65643)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65643)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_5)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_5)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65663)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N65663)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65663)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo11)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo11)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo11)
   (object    402.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo11)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#d)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#d)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion0)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo11)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo12)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65709)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65709)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo21)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo21)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo4)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo21)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo3)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo21)
   (object    200.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo21)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65738)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65738)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65738)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65738)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65738)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65754)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N65754)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65754)
   (object    0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65754)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo11)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo10)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad2)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65781)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N65781)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N65781)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65781)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65810)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N65810)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65810)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65810)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65823)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65823)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65823)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65823)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65823)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65839)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N65839)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65839)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#imports)
   (subject   http://www.isaatc.ull.es/Verdino.owl)
   (object    http://sqwrl.stanford.edu/ontologies/built-ins/3.4/sqwrl.owl)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#imports)
   (subject   http://www.isaatc.ull.es/Verdino.owl)
   (object    http://swrl.stanford.edu/ontologies/3.3/swrla.owl)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl)
   (object    http://www.w3.org/2002/07/owl#Ontology)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65878)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65878)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65878)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65878)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65878)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65894)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65894)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N65894)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65894)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo12)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo12)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo12)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo13)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo12)
   (object    402.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo12)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_4)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#e)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#e)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65937)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65937)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65937)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65937)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65937)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65969)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65969)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65982)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65982)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65982)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad1)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo7)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo9)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad1)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_16)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_16)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_16)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66019)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66019)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo22)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo22)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo4)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo22)
   (object    20.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo22)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino4)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino4)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino4)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_2)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_23)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_23)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_23)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_23)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66088)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66088)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66088)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66088)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66088)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66104)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#cardinality)
   (subject   #N66104)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N66104)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66104)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_3)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66125)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66125)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66125)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66125)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66139)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66139)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_33)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_33)
   (object    30)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_33)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_33)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66167)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66167)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66167)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66180)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66180)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66180)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66193)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66193)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#f)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#f)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66213)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66213)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66226)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66226)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo23)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo23)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo23)
   (object    20.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo23)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo23)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo10)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad4)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_17)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_17)
   (object    2)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_17)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo16)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66305)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N66305)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#cardinality)
   (subject   #N66305)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66305)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_22)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_22)
   (object    100)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_22)
   (object    100)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_22)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66334)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66334)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66334)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66334)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66347)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66347)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66347)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66347)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66347)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66363)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66363)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_15)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_15)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_17)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_15)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_16)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_15)
   (object    http://www.isaatc.ull.es/Verdino.owl#Ruta)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66405)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66405)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66405)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_2)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66425)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66425)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66425)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66425)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_32)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_32)
   (object    500)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_32)
   (object    200)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_32)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66453)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66453)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66453)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66453)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_31)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_31)
   (object    100)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_31)
   (object    100)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_31)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66481)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66481)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo10)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo10)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo10)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo12)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo10)
   (object    740.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo10)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion1)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo16)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo12)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#g)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#g)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66530)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66530)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66543)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66543)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66556)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66556)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo24)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo24)
   (object    30.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo24)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66579)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66579)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66579)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo23)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo10)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad3)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66605)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66605)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66618)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66618)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66618)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66618)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66618)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66634)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66634)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66634)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66634)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66634)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#y)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#y)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66657)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66657)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66657)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66657)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66657)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66673)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66673)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66673)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66702)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66702)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66702)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Vecindad)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vecindad)
   (object    http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vecindad)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vecindad)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Vecindad)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_25)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_25)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66751)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66751)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66751)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66751)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66751)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66767)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66767)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66767)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66780)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66780)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66780)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66793)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66793)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66793)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66793)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66793)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo0)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo0)
   (object    1.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66819)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66819)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66819)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66819)
   (object    http://www.isaatc.ull.es/Verdino.owl#m)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66819)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad6)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo24)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad6)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66848)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66848)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo15)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo15)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo19)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo15)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo17)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo15)
   (object    150.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo15)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#h)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#h)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66884)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66884)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_12)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_12)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_14)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_12)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_13)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_12)
   (object    http://www.isaatc.ull.es/Verdino.owl#Ruta)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66910)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66910)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66923)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66923)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66936)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66936)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66936)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66949)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66949)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66962)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66962)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66962)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66975)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66975)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66975)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66975)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66975)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66991)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66991)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66991)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67004)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67004)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67017)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67017)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67030)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67030)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67030)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67030)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67044)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67044)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67044)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#x)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#x)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67064)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67064)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67064)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67064)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (object    http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67094)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67094)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67094)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67107)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67107)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67120)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67120)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#i)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#i)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo22)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo13)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo11)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
   (object    150.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo16)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad5)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo14)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo16)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad5)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67172)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67172)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67172)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67172)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67185)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67185)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67185)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67198)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67198)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67198)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67198)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67198)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67214)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67214)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67227)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67227)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67227)
   (object    0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67227)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67241)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67241)
   (object    http://www.isaatc.ull.es/Verdino.owl#EnEspera)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67241)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67241)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67241)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67257)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67257)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67270)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67270)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67283)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67283)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67283)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67283)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67283)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67299)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67299)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67299)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67299)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67312)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67312)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67312)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67312)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67312)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
   (object    http://www.isaatc.ull.es/Verdino.owl#Ruta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67344)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67344)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67344)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67344)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67344)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://swrl.stanford.edu/ontologies/3.3/swrla.owl#isRuleEnabled)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-2)
   (object    true)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-2)
   (object    http://www.w3.org/2003/11/swrl#Imp)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67377)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67377)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67377)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67406)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67406)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67406)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67406)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67406)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#j)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#j)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67429)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67429)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67429)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67429)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67429)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-1)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-1)
   (object    http://www.w3.org/2003/11/swrl#Imp)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67458)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67458)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67471)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N67471)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N67471)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67471)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67484)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67484)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67497)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67497)
   (object    http://www.w3.org/2003/11/swrlb#greaterThan)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67497)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67510)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67510)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67510)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67523)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67523)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67523)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67523)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo13)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo13)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo14)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo13)
   (object    350.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo13)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67549)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67549)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67549)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67549)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67562)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67562)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67575)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67575)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67575)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67588)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67588)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67588)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67588)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67602)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67602)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67602)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67602)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67602)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#z)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#z)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Rule-3)
   (object    http://www.w3.org/2003/11/swrl#Imp)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67638)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67638)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67638)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67638)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67638)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67654)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67654)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67654)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_26)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_26)
   (object    500)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_26)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_26)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67682)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67682)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#k)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#k)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67702)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67702)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67702)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67702)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad7)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo22)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad7)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67728)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67728)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67728)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67728)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67741)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67741)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67741)
   (object    http://www.isaatc.ull.es/Verdino.owl#EnEspera)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67741)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67741)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67757)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67757)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67770)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67770)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (object    http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67799)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N67799)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N67799)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67799)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67812)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67812)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67812)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67825)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67825)
   (object    http://www.w3.org/2003/11/swrlb#greaterThan)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67825)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67838)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67838)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67838)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67838)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67838)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo14)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo14)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo14)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo11)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo14)
   (object    350.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo14)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67870)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67870)
   (object    http://www.w3.org/2003/11/swrlb#equal)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67870)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67899)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N67899)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N67899)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67899)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria_8)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria_8)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67935)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67935)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67935)
   (object    http://www.isaatc.ull.es/Verdino.owl#EnEspera)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67935)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67935)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67951)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67951)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo9)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo5)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo4)
   (object    490.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67980)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67980)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_34)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino)
   (object    10.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68011)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68011)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68011)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68024)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68024)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68037)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68037)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68037)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68050)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68050)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_29)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_29)
   (object    500)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_29)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_29)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68078)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68078)
   (object    http://www.isaatc.ull.es/Verdino.owl#n)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68078)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68091)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68091)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68091)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68091)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68091)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68107)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N68107)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#minCardinality)
   (subject   #N68107)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68107)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#m)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#m)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo3)
   (object    290.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68138)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68138)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68138)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68151)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68151)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68164)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68164)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo19)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo19)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo20)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo19)
   (object    350.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo19)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68190)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N68190)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68190)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68190)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68203)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68203)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68216)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68216)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68216)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68216)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vecindad)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
   (object    http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68245)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68245)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68258)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68258)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68258)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68271)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68271)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion7)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo17)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68297)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68297)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68310)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68310)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68339)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68339)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68339)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68339)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68353)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68353)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68353)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo23)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo2)
   (object    200.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68379)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68379)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68379)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo18)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo18)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo12)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo18)
   (object    390.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo18)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#n)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#n)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68412)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68412)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68412)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68412)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68412)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68428)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68428)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68428)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68428)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68428)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#EnEspera)
   (object    http://www.isaatc.ull.es/Verdino.owl#Estado)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68451)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68451)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68451)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68451)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68451)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68467)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68467)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68467)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68496)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68496)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68496)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68496)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68496)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68512)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68512)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68512)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68512)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68512)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
   (object    http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#disjointWith)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vecindad)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68560)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68560)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68586)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68586)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68586)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68586)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68586)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#o)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#o)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68609)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68609)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68609)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo24)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo22)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo2)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
   (object    260.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo1)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo17)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo17)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo18)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo17)
   (object    390.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo17)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68654)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68654)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68654)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68667)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68667)
   (object    http://www.isaatc.ull.es/Verdino.owl#n)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68667)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68680)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68680)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68700)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68700)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68713)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68713)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68726)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68726)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_10)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_10)
   (object    180.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_10)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion5)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo20)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68763)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68763)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68763)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68763)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68763)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#c)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#c)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo8)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo8)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo10)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo8)
   (object    200.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo8)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68799)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68799)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68799)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68799)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68799)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad0)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo25)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad0)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo21)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Prioridad0)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion6)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo18)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#b)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#b)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo9)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo9)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo8)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo9)
   (object    250.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo9)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo7)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo8)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo7)
   (object    290.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo7)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68874)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N68874)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#maxCardinality)
   (subject   #N68874)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68874)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_12)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_12)
   (object    180.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_12)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
   (object    http://www.isaatc.ull.es/Verdino.owl#Estado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino3)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_10)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino3)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino3)
   (object    1.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_13)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_13)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_13)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68957)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68957)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#RelacionEntreTramos)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#a)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#a)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_2)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion3)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo12)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo16)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion3)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_7)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_7)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#inverseOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePredecesor)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69036)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69036)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69036)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69036)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69036)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo6)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo7)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo6)
   (object    250.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo6)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_30)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_30)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_30)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica_30)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionGrafica)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (object    http://www.isaatc.ull.es/Verdino.owl#Ruta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#tienePredecesor)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#inverseOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePredecesor)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tienePredecesor)
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo5)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo6)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo5)
   (object    290.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo5)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino2)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino2)
   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_12)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino2)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneAceleracion)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino2)
   (object    0.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Verdino2)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo20)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneSucesor)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo20)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo16)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo20)
   (object    350.0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Tramo20)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69163)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69163)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69163)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69163)
   (object    http://www.isaatc.ull.es/Verdino.owl#m)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69163)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_14)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_14)
   (object    2)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden_14)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_6)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Variable_6)
   (object    http://www.w3.org/2003/11/swrl#Variable)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion4)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo19)
  )
)
  
(assert
  (triple
   (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo15)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion4)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#Estado)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Estado)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69217)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69217)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#subClassOf)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
   (object    http://www.w3.org/2002/07/owl#Thing)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#TramoOrden)
   (object    http://www.w3.org/2002/07/owl#Class)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69246)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N69246)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69246)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69246)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  

  ;;; ------------------- Run Jess Rule Engine ------------------------

(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
;(run)
(save-facts owlfacts.txt)

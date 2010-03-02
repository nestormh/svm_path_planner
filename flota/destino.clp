

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
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65551)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65551)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65551)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65551)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N65591)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65591)
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
    (subject   #N65673)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65673)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaFinal)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65673)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65673)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65673)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65689)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65689)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N65689)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65689)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65718)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65718)
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
    (subject   #N65773)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65773)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65816)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65816)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65816)
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
    (subject   #N65863)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65863)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65863)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65863)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65863)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N65892)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65892)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N65923)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65923)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N65951)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N65951)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N65951)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N65951)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65951)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaDistancia)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaDistancia)
   (object    http://www.isaatc.ull.es/Verdino.owl#Estado)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N65981)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N65981)
   (object    http://www.isaatc.ull.es/Verdino.owl#n)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N65981)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
    (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaOposicion)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaOposicion)
   (object    http://www.isaatc.ull.es/Verdino.owl#Estado)
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
    (subject   #N66023)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66023)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66023)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
    (subject   #N66117)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66117)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66130)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66130)
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
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_16)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
    (subject   #N66194)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66194)
   (object    http://www.w3.org/2003/11/swrlb#equal)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66194)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   #N66258)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66258)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66258)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   #N66294)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66294)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N66314)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66314)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66314)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66314)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66327)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N66327)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N66327)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66327)
   (object    http://www.w3.org/2002/07/owl#Restriction)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
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
    (subject   #N66392)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66392)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66392)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66392)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66405)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N66428)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66428)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66428)
   (object    http://www.isaatc.ull.es/Verdino.owl#m)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66428)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66428)
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
    (subject   #N66473)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66473)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66486)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66486)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66486)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   #N66506)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66506)
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
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoOrden_13)
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
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Ruta_12)
   (object    http://www.isaatc.ull.es/Verdino.owl#Ruta)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66532)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N66532)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N66532)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66532)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66545)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66545)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66545)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66545)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66558)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66558)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66558)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
    (subject   #N66578)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66578)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66578)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66591)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66591)
   (object    http://www.isaatc.ull.es/Verdino.owl#EsperaDistancia)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66591)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66591)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66591)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
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
    (subject   #N66630)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66630)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N66675)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66675)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66675)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66688)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66688)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N66688)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66688)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
   (predicate http://www.w3.org/2002/07/owl#minCardinality)
   (subject   #N66702)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N66702)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66702)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66716)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66716)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66729)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66729)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66729)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66742)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N66742)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66742)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66771)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66771)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66771)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66771)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66771)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66787)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66787)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66787)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66787)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66787)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66803)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66803)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N66833)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66833)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
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
    (subject   #N66869)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66869)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#range)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
   (object    http://www.w3.org/2001/XMLSchema#float)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2000/01/rdf-schema#domain)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
    (subject   #N66924)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N66924)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66924)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66937)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66937)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66950)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N66950)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66950)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N66950)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66950)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66966)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N66966)
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N66966)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66966)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N66979)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66979)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N66999)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N66999)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N67025)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67025)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67038)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67038)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67038)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67038)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67038)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67054)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67054)
   (object    http://www.isaatc.ull.es/Verdino.owl#n)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67054)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67067)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67067)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N67095)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N67095)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N67095)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67095)
   (object    http://www.w3.org/2002/07/owl#Restriction)
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
    (subject   #N67115)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67115)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67157)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67157)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67170)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67170)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67170)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67170)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67170)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#ObjectProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67218)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67218)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67218)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67218)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67231)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67231)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67244)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67244)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67244)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67244)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
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
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67257)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
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
    (subject   #N67277)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67277)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67277)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67306)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67306)
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
    (subject   #N67353)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67353)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaInterseccionPrioritaria)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#EsperaInterseccionPrioritaria)
   (object    http://www.isaatc.ull.es/Verdino.owl#Estado)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67373)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67373)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67386)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67386)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67399)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67399)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67412)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67412)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67412)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67425)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67425)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67438)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67438)
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
    (subject   #N67466)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67466)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67466)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67479)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67479)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67479)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67479)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67479)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67495)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67495)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67495)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67495)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoPrioritario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67495)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67511)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67511)
   (object    http://www.w3.org/2003/11/swrlb#greaterThan)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67511)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67524)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67524)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67537)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67537)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N67580)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67580)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67580)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67580)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67593)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67593)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67606)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67606)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67606)
   (object    http://www.isaatc.ull.es/Verdino.owl#EsperaInterseccionPrioritaria)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67606)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67606)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67622)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67622)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67622)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67622)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67635)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67635)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67635)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67635)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67635)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67651)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67651)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67651)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67664)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N67664)
   (object    http://www.isaatc.ull.es/Verdino.owl#TramoPintable)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67664)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67664)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67677)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67677)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67677)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67677)
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
   (object    http://www.isaatc.ull.es/Verdino.owl#Oposicion)
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
    (subject   #N67707)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67707)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67707)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67707)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67707)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67723)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N67723)
   (object    http://www.w3.org/2003/11/swrlb#lessThanOrEqual)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67723)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67736)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67736)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67736)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67736)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67736)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67752)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67752)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67752)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
    (subject   #N67778)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67778)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67791)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67791)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67791)
   (object    http://www.isaatc.ull.es/Verdino.owl#o)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67791)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67791)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
    (subject   #N67836)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67836)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67836)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67836)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67836)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N67872)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67872)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67872)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67872)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67885)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67885)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67898)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N67898)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaY)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N67898)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67898)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67911)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67911)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67924)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67924)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67924)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67937)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67937)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67937)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N67957)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N67957)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N67957)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N67957)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67957)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67973)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N67973)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67973)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67986)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N67986)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67986)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N67999)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N67999)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68012)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68012)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68025)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68025)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68025)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   #N68054)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68054)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68067)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68067)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68067)
   (object    50)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68067)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
   (object    http://www.w3.org/2002/07/owl#FunctionalProperty)
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
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68097)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68097)
   (object    0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68097)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68097)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68111)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68111)
   (object    http://www.isaatc.ull.es/Verdino.owl#m)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68111)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68111)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68111)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
   (object    http://www.isaatc.ull.es/Verdino.owl#Vecindad)
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
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   http://www.isaatc.ull.es/Verdino.owl#Oposicion)
   (object    http://www.w3.org/2002/07/owl#Class)
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
    (subject   #N68156)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68156)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68169)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68169)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N68208)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68208)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68208)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68221)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68221)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68221)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68221)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
    (subject   #N68248)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68248)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68248)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68261)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68261)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68274)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68274)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68274)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68274)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68274)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N68297)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68297)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68297)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68297)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68297)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68313)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68313)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68313)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68326)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68326)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionGraficaInicial)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68326)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68326)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68326)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68342)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68342)
   (object    300)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68342)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68342)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68356)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68356)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68356)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68356)
   (object    http://www.isaatc.ull.es/Verdino.owl#i)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68356)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68372)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68372)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68372)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68372)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68385)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#cardinality)
   (subject   #N68385)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N68385)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68385)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68399)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68399)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68399)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68399)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68399)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68415)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68415)
   (object    http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68415)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68415)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68415)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N68455)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68455)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N68475)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68475)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68475)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68475)
   (object    http://www.isaatc.ull.es/Verdino.owl#k)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68475)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68491)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68491)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68504)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68504)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68504)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68517)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68517)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68517)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
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
    (subject   #N68543)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68543)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68543)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68556)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68556)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68556)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68556)
   (object    http://www.isaatc.ull.es/Verdino.owl#g)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68556)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68572)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N68572)
   (object    http://www.isaatc.ull.es/Verdino.owl#InterseccionPrioritaria)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68572)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68572)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68585)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68585)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68598)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#someValuesFrom)
   (subject   #N68598)
   (object    http://www.w3.org/2001/XMLSchema#int)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N68598)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneCoordenadaX)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68598)
   (object    http://www.w3.org/2002/07/owl#Restriction)
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
    (subject   #N68624)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68624)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68637)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68637)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68637)
   (object    http://www.isaatc.ull.es/Verdino.owl#j)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68637)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68637)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N68673)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68673)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68673)
   (object    http://www.isaatc.ull.es/Verdino.owl#a)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68673)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68673)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68689)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N68689)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68689)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68702)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68702)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N68741)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N68741)
   (object    http://www.w3.org/2003/11/swrlb#greaterThan)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68741)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   #N68778)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68778)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68778)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68778)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68778)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
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
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
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
    (subject   #N68839)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68839)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68839)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68839)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68839)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68855)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#maxCardinality)
   (subject   #N68855)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N68855)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneRuta)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68855)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68869)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68869)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68869)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68869)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoSecundario)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68869)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68885)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68885)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N68885)
   (object    http://www.isaatc.ull.es/Verdino.owl#Tramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68885)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
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
    (subject   #N68914)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N68914)
   (object    http://www.isaatc.ull.es/Verdino.owl#y)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N68914)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N68914)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68914)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68930)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68930)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N68977)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68977)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N68990)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N68990)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N68990)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69003)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#builtin)
   (subject   #N69003)
   (object    http://www.w3.org/2003/11/swrlb#subtract)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69003)
   (object    http://www.w3.org/2003/11/swrl#BuiltinAtom)
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
    (subject   http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
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
   (object    http://www.w3.org/2002/07/owl#DatatypeProperty)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69048)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69048)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69061)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69061)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69074)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N69074)
   (object    http://www.isaatc.ull.es/Verdino.owl#e)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69074)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69087)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69087)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N69151)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#onProperty)
   (subject   #N69151)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2002/07/owl#cardinality)
   (subject   #N69151)
   (object    1)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69151)
   (object    http://www.w3.org/2002/07/owl#Restriction)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69165)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69165)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#classPredicate)
   (subject   #N69165)
   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69165)
   (object    http://www.w3.org/2003/11/swrl#ClassAtom)
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
    (subject   #N69222)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69222)
   (object    http://www.w3.org/2003/11/swrl#AtomList)
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
    (subject   #N69246)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69246)
   (object    http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69246)
   (object    http://www.isaatc.ull.es/Verdino.owl#z)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69246)
   (object    http://www.isaatc.ull.es/Verdino.owl#c)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69246)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69262)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69262)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69262)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69262)
   (object    http://www.isaatc.ull.es/Verdino.owl#f)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69262)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69278)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69278)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69278)
   (object    http://www.isaatc.ull.es/Verdino.owl#h)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69278)
   (object    http://www.isaatc.ull.es/Verdino.owl#d)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69278)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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
    (subject   #N69314)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69314)
   (object    http://www.isaatc.ull.es/Verdino.owl#b)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69314)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69314)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneDistanciaAConflicto)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69314)
   (object    http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom)
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
    (subject   #N69337)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#rest)
   (subject   #N69337)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#nil)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#first)
   (subject   #N69337)
   (object    0)
  )
)  
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69337)
   (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#List)
  )
)
  
  
(assert
  (triple
    (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
    (subject   #N69351)
    (object    http://www.w3.org/1999/02/22-rdf-syntax-ns#Description)
  )
)
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument2)
   (subject   #N69351)
   (object    http://www.isaatc.ull.es/Verdino.owl#EsperaOposicion)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#argument1)
   (subject   #N69351)
   (object    http://www.isaatc.ull.es/Verdino.owl#x)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/2003/11/swrl#propertyPredicate)
   (subject   #N69351)
   (object    http://www.isaatc.ull.es/Verdino.owl#tieneEstado)
  )
)
  
(assert
  (triple
   (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type)
   (subject   #N69351)
   (object    http://www.w3.org/2003/11/swrl#IndividualPropertyAtom)
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

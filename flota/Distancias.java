

import jess.*;
import java.util.*;
import java.io.*;
import com.hp.hpl.jena.ontology.*;
import com.hp.hpl.jena.rdf.model.ModelFactory;
import com.hp.hpl.jena.rdf.model.*;

import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;


//OJO QUE ESTAS HACIENDO LAS BUSQUEDAS EN JESS

public class Distancias {

Rete engine = new Rete();
Rete engineRutas = new Rete();
Hashtable hashVehiculos = new Hashtable();
//static String assertLimpiezaEstados = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarRuta) ))";
	String prefijo="http://www.isaatc.ull.es/Verdino.owl#";
	OntModel m = ModelFactory.createOntologyModel( OntModelSpec.OWL_MEM, null );
static String assertLimpiezaPosiciones = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarPosicionesAntiguas) ))";
static String assertLimpiezaTramos = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarTramosAntiguos) ))";
static String assertLimpiezaRutas = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarRutas) ))";

	static String assertLimpiezaEstados = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarEstado) ))";
static String assertLimpiezaVelocidades = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#Accion) (subject   http://www.isaatc.ull.es/Verdino.owl#AccionActual) (object   http://www.isaatc.ull.es/Verdino.owl#LimpiarVelocidadesAntiguas) ))";

	
    public static void main(String[] argv) throws JessException ,  java.io.IOException,TransformerException, TransformerConfigurationException, 
    FileNotFoundException {
        String origen = "Tramo" + argv[0];
		String destino = "Tramo" + argv[1];
		
		Distancias d = new Distancias(origen, destino);
	}
	
	public Distancias() throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException, 
    FileNotFoundException
	{ //definicionesTramos();
	  //transformaHechosJess();
	   //transformaReglasJess();
	   //arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
	   //cargarHechosReglas();
    //	calculaRuta();
	}
	
	public Distancias(String origen, String destino) throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException, 
    FileNotFoundException
	{   // hacemos copias locales
	    cargaEspaciosDeNombres();
		definicionesTramos();
		transformaHechosJess();
		transformaReglasJess();
		arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		cargarHechosReglas();
	// assert tramos (leerlos de ontologias) y preparar la ontología del cálculo de rutas
		leerTramosDeOntologia();  
	// leer los vehiculos de la ontologia y crear los objetos.  
	   leerVehiculos();
		Vector ruta = calculaRuta(origen, destino);
	 // en caso de quererlo asignar a un vehiculo
	 // fijarRutaVehiculo(id, ruta);
	// poner el sistema en marcha
	  funcionamiento();
	}
	

	public void cargaEspaciosDeNombres()
	{System.out.println("LEYENDO VERDINO");
        m.getDocumentManager().addAltEntry( "http://www.isaatc.ull.es/Verdino.owl","file:verdino.owl" );
		m.getDocumentManager().addAltEntry( "http://swrl.stanford.edu/ontologies/3.3/swrla.owl",
                                          "file:swrla.owl.xml" );
		m.getDocumentManager().addAltEntry( "http://sqwrl.stanford.edu/ontologies/built-ins/3.4/sqwrl.owl",
                                          "file:sqwrl.owl.xml" );							
		m.getDocumentManager().addAltEntry( "http://protege.stanford.edu/plugins/owl/protege",
                                          "file:protege.owl" );							
		m.read( "http://www.isaatc.ull.es/Verdino.owl" );
	}
	
	public void limpiaRutas() throws JessException
	{engine.executeCommand(assertLimpiezaRutas);
	 engine.run();
	}
	
	public void fijarRutaVehiculo(String idVehiculo, String[] ruta)
	{// poner el prefijo
	 String[] rut = new String[ruta.length];
	 for (int i=0; i < ruta.length; i++)
	 {rut[i] = prefijo + ruta[i];
	 }
	 Vehiculo actual = (Vehiculo) hashVehiculos.get(idVehiculo);
	 actual.fijaRuta(rut);
	}
	
	
	public void fijarRutaVehiculo(String idVehiculo, Vector ruta)
	{// poner el prefijo
	 System.out.println("Fijar ruta vehiculo");
	 String[] rutaString = new String[ruta.size()];
	 for (int i=0; i < ruta.size(); i++)
	 {rutaString[i] = (String) ruta.elementAt(i);
	 }
	 Vehiculo actual = (Vehiculo) hashVehiculos.get(idVehiculo);
	 actual.fijaRuta(rutaString);
	}
	
	public void leerTramosDeOntologia() throws JessException
	{ System.out.println("leer tramos de ontologia");
	engineRutas = new Rete();
	 engineRutas.reset();
	 engineRutas.clear();
	 engineRutas.executeCommand("(batch factsdistancias.clp)");
	 engineRutas.executeCommand("(batch distancias.clp)");
	 Vector vector = calculaVectorTramosDeOntologia();
	 assertsTramos(vector);
	}
	 
public void tramosAEngineRutas() throws JessException
	{ System.out.println("tramos a engineRutas");
	engineRutas = new Rete();
	 engineRutas.reset();
	 engineRutas.clear();
	 engineRutas.executeCommand("(batch factsdistancias.clp)");
	 engineRutas.executeCommand("(batch distancias.clp)");
	// Vector vector = calculaVectorTramosDeOntologia();
	// assertsTramos(vector);
	}
		
	
	 
	public Vector calculaVectorTramosDeOntologia() throws JessException
   { System.out.println("calculaVectorTramosDeOntologia");
		Vector tramos = new Vector();
		ValueVector vector = new ValueVector();
		String aver = "http://www.isaatc.ull.es/Verdino.owl#Tramo";
		vector.add(new Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", RU.ATOM));
		vector.add(new Value(aver,RU.ATOM));
        Iterator result2 = engine.runQuery("buscaSujetosConObjeto", vector);
        String estado = " ";
		while (result2.hasNext()) {
			 Token t = (Token) result2.next();
             Fact f = (Fact) t.fact(1);
			String id = f.getSlotValue("subject").stringValue(engine.getGlobalContext());
            //System.out.println("TRAMO = " + id);
			Tramo actual = new Tramo(id);
			actual.fijaLongitud(dimeLongitudTramo(id));
			actual.fijaVectorNombresSucesores(dimeSucesoresTramo(id));
			tramos.addElement(actual);
		 	}
		return tramos;
		}    
		
		public String quitaPrefijo(String principio)
		{int longitudPrefijo = prefijo.length();
		 String solucion = principio.substring(longitudPrefijo);
		 return solucion;
		}
	 	 	 	 
	public void leerVehiculos() throws JessException
	{System.out.println("leer vehiculo");
	ValueVector vector = new ValueVector();
	String aver = "http://www.isaatc.ull.es/Verdino.owl#Vehiculo";
	vector.add(new Value("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", RU.ATOM));
	vector.add(new Value(aver,RU.ATOM));
    Iterator result2 = engine.runQuery("buscaSujetosConObjeto", vector);
		while (result2.hasNext()) {
			 Token t = (Token) result2.next();
		      Fact f = (Fact) t.fact(1);
			String id = f.getSlotValue("subject").stringValue(engine.getGlobalContext());
			      System.out.println("VEHICULO = " + id);
		Vehiculo vehiculo = new Vehiculo(id);
			hashVehiculos.put(id, vehiculo);
			String sujeto = id;
			    Individual individuo1 = m.getIndividual(id);
	  ObjectProperty tienePosicionVehiculo = m.getObjectProperty(prefijo + "tienePosicionVehiculo");
      NodeIterator it = individuo1.listPropertyValues(tienePosicionVehiculo);
	  String posicion = "";
	  while(it.hasNext())
	  {posicion = (String) it.next().toString();
	   }	  
	  vehiculo.fijaPosicionVehiculo(posicion);
	   }
	   	engine.executeCommand(assertLimpiezaPosiciones);
		inicializacionFicticia();  //método a quitar en situaciones reales
	}
	
	public void inicializaVehiculos(String[] idVehiculos) throws JessException
	{for (int i=0; i<idVehiculos.length; i++)
	 {String id = prefijo + idVehiculos[i];
	  String a1 = "(assert (triple (predicate http://www.w3.org/1999/02/22-rdf-syntax-ns#type) (subject   http://www.isaatc.ull.es/Verdino.owl#" + idVehiculos[i] +")   (object    http://www.isaatc.ull.es/Verdino.owl#Vehiculo)))";
      engine.executeCommand(a1);
	  String posicion = prefijo + "PosicionVehiculo_" + idVehiculos[i];
	  String a2 = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tienePosicionVehiculo) (subject   http://www.isaatc.ull.es/Verdino.owl#" + idVehiculos[i] +")   (object    http://www.isaatc.ull.es/Verdino.owl#PosicionVehiculo_" + idVehiculos[i] + ")))";
      engine.executeCommand(a2);
	  Vehiculo vehiculo = new Vehiculo(id);
	  hashVehiculos.put(id, vehiculo);
	  vehiculo.fijaPosicionVehiculo(posicion);
	 }
	 engine.executeCommand(assertLimpiezaPosiciones);
	}
	
	public void asignaRuta(String idVehiculo, String[] ruta) throws JessException
	{ String idVehiculos = prefijo + idVehiculo;
	  Vehiculo v1 = (Vehiculo) hashVehiculos.get(idVehiculos);
	  UUID idOne = UUID.randomUUID();
      //System.out.println("ID=" + idOne);
	String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta) (subject ";
	 stringAssert = stringAssert + idVehiculos +  ") (object " + prefijo +  idOne +" ) ) ) "; 
	   engine.executeCommand(stringAssert);
	 fijarRutaVehiculo(idVehiculos, ruta);
	for (int i=0; i<ruta.length; i++)
	 {stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden) (subject ";
	 stringAssert = stringAssert + prefijo + idOne +  ") (object " + prefijo +  "TramoOrden1_" + idOne + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	   stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden1_" + idOne +  ") (object " + (i+1) + ") ) ) "; 
	 	   engine.executeCommand(stringAssert);
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden1_" + idOne +  ") (object "  + prefijo +  ruta[i] + ") ) ) "; 
	//System.out.println(ruta[i]);
	   engine.executeCommand(stringAssert);
	 }
    
	}
	
	public void inicializacionFicticia() throws JessException
	{System.out.println("INICIALIZACION FICTICIA-----------------");
	engine.executeCommand(assertLimpiezaPosiciones);
	 String id = prefijo + "Verdino";
	 Vehiculo v1 = (Vehiculo) hashVehiculos.get(id);
	 v1.fijaVelocidad(10);
	 v1.fijaTramo(prefijo + "Tramo10");
	 v1.fijaLongitudEnTramo(620);
	 System.out.println("INICIALIZACION FICTICIA" + v1);
	 String[] ruta1 = {"Tramo10", "Tramo12", "Tramo15", "Tramo19"};
     String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta) (subject ";
	 stringAssert = stringAssert + id +  ") (object " + prefijo +  "Ruta1) ) ) "; 
	   engine.executeCommand(stringAssert);
	 fijarRutaVehiculo(id, ruta1);
	 id= prefijo + "Verdino2";
	 Vehiculo v2 = (Vehiculo) hashVehiculos.get(id);
	 v2.fijaVelocidad(10);
	 v2.fijaTramo(prefijo + "Tramo14");
	 v2.fijaLongitudEnTramo(310);
	 String[] ruta2 = {"Tramo14", "Tramo15", "Tramo19", "Tramo20"};
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta) (subject ";
	 stringAssert = stringAssert + id +  ") (object " + prefijo +  "Ruta2) ) ) "; 
	   engine.executeCommand(stringAssert);
	 fijarRutaVehiculo(id, ruta2);
	 id= prefijo + "Verdino3";
	 Vehiculo v3 = (Vehiculo) hashVehiculos.get(id);
	 v3.fijaVelocidad(15);
	 v3.fijaTramo(prefijo + "Tramo16");
	 v3.fijaLongitudEnTramo(90);
	 String[] ruta3 = {"Tramo16", "Tramo11", "Tramo21", "Tramo4"};
	   stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta) (subject ";
	 stringAssert = stringAssert + id +  ") (object " + prefijo +  "Ruta3) ) ) "; 
	   engine.executeCommand(stringAssert);
	 fijarRutaVehiculo(id, ruta3);
	 id= prefijo + "Verdino4";
	 Vehiculo v4 = (Vehiculo) hashVehiculos.get(id);
	 v4.fijaVelocidad(5);
	 v4.fijaTramo(prefijo + "Tramo1");
	 v4.fijaLongitudEnTramo(55);
	 String[] ruta4 = {"Tramo1", "Tramo24", "Tramo22", "Tramo4", "Tramo5"};
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneRuta) (subject ";
	 stringAssert = stringAssert + id +  ") (object " + prefijo +  "Ruta4) ) ) "; 
	   engine.executeCommand(stringAssert);
	 fijarRutaVehiculo(id, ruta4);
	for (int i=0; i<ruta1.length; i++)
	 {stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "Ruta1" +  ") (object " + prefijo +  "TramoOrden1_" + i + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	   stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden1_" + i +  ") (object " + (i+1) + ") ) ) "; 
	 	   engine.executeCommand(stringAssert);
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden1_" + i +  ") (object "  + prefijo + ruta1[i] + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	 }
     for (int i=0; i<ruta2.length; i++)
	 {stringAssert = "(assert (triple (predicate " + prefijo + "tieneTramoOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "Ruta2" +  ") (object " + prefijo +  "TramoOrden2_" + i + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	     stringAssert = "(assert (triple (predicate " + prefijo + "tieneOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden2_" + i +  ") (object " + (i+1) + ") ) ) "; 
	 	   engine.executeCommand(stringAssert);
	  stringAssert = "(assert (triple (predicate " + prefijo + "tieneTramo) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden2_" + i +  ") (object "  + prefijo +  ruta2[i] + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	 }
	  for (int i=0; i<ruta3.length; i++)
	 {stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramoOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "Ruta3" +  ") (object " + prefijo +  "TramoOrden3_" + i + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	     stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneOrden) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden3_" + i +  ") (object " + (i+1) + ") ) ) "; 
	  	   engine.executeCommand(stringAssert);
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneTramo) (subject ";
	 stringAssert = stringAssert + prefijo + "TramoOrden3_" + i +  ") (object "  + prefijo +  ruta3[i] + ") ) ) "; 
	   engine.executeCommand(stringAssert);
	 }
    }
	
	
	//método a cambiar cuando funcione el GPS... en realidad, la clase vehiculo
	public double dimeVelocidadVehiculo (String id) throws JessException
	{System.out.println("dimevelocidadvehiculo");
	Vehiculo actual = (Vehiculo) hashVehiculos.get(id);
	 return actual.dimeVelocidad();
	}
	
		//método a cambiar cuando funcione el GPS
	public String dimeTramoVehiculo(String id) throws JessException
	{System.out.println("dime tramo vehiculo");
	Vehiculo actual = (Vehiculo) hashVehiculos.get(id);
	 return actual.dimeTramo();
	}
	
	//método a cambiar cuando funcione el GPS
	public double dimePosicionEnTramoVehiculo(String id) throws JessException
	{System.out.println("dime posicion en tramo");
	Vehiculo actual = (Vehiculo) hashVehiculos.get(id);
	 return actual.dimeLongitudEnTramo();
	}
	
	
	public double dimeLongitudTramo(String tramo) throws JessException
	{ // buscar longitud tramo actual en ontologia			
//System.out.println("dime longitud en tramo");
 ValueVector vector = new ValueVector();
 vector.add(new Value("http://www.isaatc.ull.es/Verdino.owl#tieneLongitud", RU.ATOM));
		vector.add(new Value(tramo,RU.ATOM));
        Iterator result2 =
            engine.runQuery("buscaObjetosConSujeto", vector);
    		double longitudTramo = 0;
  		while (result2.hasNext()) {
			 Token t = (Token) result2.next();
             Fact f = (Fact) t.fact(1);
			String id = f.getSlotValue("object").stringValue(engine.getGlobalContext());
			longitudTramo = (double) Double.parseDouble(id);
			}
			return longitudTramo;
	}
	
	
	public void arreglarReglas(String openFile, String saveFile)
  {System.out.println("arreglar reglas");
   String linea;
  if((openFile)!= null) 
   {try
    {BufferedReader data = new BufferedReader (new FileReader(openFile));
	  PrintWriter ficheroSalida = new PrintWriter(new BufferedWriter ( new FileWriter(saveFile)));
      Vector vector = new Vector();
	  Vector vectorBind = new Vector();
	  Vector vectorBinded = new Vector();
	  Vector vectorTriples = new Vector();
	  Vector vectorTriplesSencillos = new Vector();
	  boolean antecedentes = true;
	  while(true)
	 {linea = data.readLine();
	  if (linea == null)
	  {break;
	  }
	  String lineaTrimeada = linea.trim();
	  if (lineaTrimeada.startsWith("(defrule"))
	  {antecedentes = true;
	  }
	  if (lineaTrimeada.startsWith("(test ") || lineaTrimeada.startsWith("(bind ") || lineaTrimeada.startsWith("=>"))
	  {if (lineaTrimeada.startsWith("(test "))
	   {vector.addElement(lineaTrimeada);
	   }
	   if (lineaTrimeada.startsWith("(bind "))
	   {int indice = lineaTrimeada.indexOf("?");
	    lineaTrimeada = lineaTrimeada.substring(indice, lineaTrimeada.length());
		indice = lineaTrimeada.indexOf(" ");
		String variable1 = lineaTrimeada.substring(0, indice);
		lineaTrimeada = lineaTrimeada.substring(indice, lineaTrimeada.length() - 1);
		vectorBind.addElement(lineaTrimeada);
		vectorBinded.addElement(variable1);
	   }
	   if (lineaTrimeada.startsWith("=>"))
	   {antecedentes = false;
	    Vector vectorVariables = new Vector();
	    for (int i=0; i<vectorTriplesSencillos.size(); i++)
		{ficheroSalida.println(vectorTriplesSencillos.elementAt(i));
		 String triples = (String)(vectorTriplesSencillos.elementAt(i));
		 int indice = triples.indexOf("?");
		 String aux = triples.substring(indice, triples.length());
		 indice = aux.indexOf(")");
		 aux = aux.substring(0,indice+1);
		 ficheroSalida.println(";" + aux);
		 vectorVariables.addElement(aux);
		 
		 }
		for (int i=0; i<vectorTriples.size(); i++)
		{String cuerda = (String)(vectorTriples.elementAt(i));
		 for (int j=0; j<vectorVariables.size(); j++)
		 {String variable = (String) vectorVariables.elementAt(j);
		  int indice = cuerda.indexOf(variable);
		  if(indice>0)
		  {ficheroSalida.println(vectorTriples.elementAt(i));
		   vectorTriples.removeElementAt(i);
		   i--;
		  }
		 }
		}
		for (int i=0; i<vectorTriples.size(); i++)
		{ficheroSalida.println(vectorTriples.elementAt(i));
		}
		for (int i=0; i<vector.size(); i++)
		{String inicial = (String)vector.elementAt(i);
		 for (int k= 0; k < vectorBind.size(); k++)
		 {String reemplazado = (String)vectorBinded.elementAt(k);
		  String reemplazo = (String)vectorBind.elementAt(k);
		  inicial = inicial.replace(reemplazado+ " ", reemplazo + " ");
		 }
		 ficheroSalida.println(inicial);
		}
		ficheroSalida.println ("=>");
		vectorBind.removeAllElements();
		vectorBinded.removeAllElements();
		vector.removeAllElements();
		vectorTriplesSencillos.removeAllElements();
		vectorTriples.removeAllElements();
		}
	  }
	  else 
	  {if (lineaTrimeada.startsWith("(triple") && antecedentes)
	   {String triple = lineaTrimeada;
	    triple = triple + "\n" + data.readLine();
		triple = triple + "\n" + data.readLine();
		triple = triple + "\n" + data.readLine();
		triple = triple + "\n" + data.readLine();
		int numeroVariable = 0;
		int indice = triple.indexOf("?");
		if(indice > 0)
		{numeroVariable++;
		 String aux = triple;
		aux = aux.substring(indice+1, aux.length());
		 indice = aux.indexOf("?");
		 if(indice > 0)
		{numeroVariable++;
		aux = aux.substring(indice+1, aux.length());
		 indice = aux.indexOf("?");
		 if(indice > 0)
		{numeroVariable++;
		aux = aux.substring(indice+1, aux.length());
		 indice = aux.indexOf("?");
		}
		 }
		}
		if (numeroVariable ==1)
		{vectorTriplesSencillos.addElement(triple);
		}
		else
		{vectorTriples.addElement(triple);
		}
	   }
	   else
	   {ficheroSalida.println(lineaTrimeada);
	   }
	  }
	 }
	 data.close();
	 ficheroSalida.close();
   } catch (IOException exc) {}
  }
  }
	
	public void cargarHechosReglas() throws JessException
	{System.out.println("cargar hechos reglas");
	 engine.reset();
	 engine.clear();
	 engine.executeCommand("(batch destino.clp)");  // los hechos: tramos, definiciones, etc..
	 engine.executeCommand("(batch reglasarregladas.clp)");  // las reglas
	 engine.executeCommand("(batch consultasVerdino.clp)");  // las consultas
     engine.executeCommand("(batch scriptsadicionales.clp)");  // script adicional
	}
	
	public void transformaReglasJess() throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException, 
    FileNotFoundException
	{System.out.println("Transformando reglas");
	    String fileName = "verdino"; // an SWRL ontology
		String xmlFileName = fileName+".owl";
    	String xslFileName = "SWRL2Jessrevisada.xsl.xml";
    	String outOFxslFileName = "reglasverdino"+".clp";
    	TransformerFactory tFactory = TransformerFactory.newInstance();
    	Transformer transformer = tFactory.newTransformer(new StreamSource(xslFileName));
    	transformer.transform(new StreamSource(xmlFileName), new StreamResult(new FileOutputStream(outOFxslFileName)));
    	}
	
	
	public void transformaHechosJess() throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException, 
    FileNotFoundException
	{System.out.println("Transformando hechos");
	    String fileName = "destino";
		String xmlFileName = fileName+".owl";
    	String xslFileName = "OWL2Jessrevisada.xsl.xml";
    	String outOFxslFileName = fileName+".clp";
   	   	TransformerFactory tFactory = TransformerFactory.newInstance();
    	Transformer transformer = tFactory.newTransformer(new StreamSource(xslFileName));
    	transformer.transform(new StreamSource(xmlFileName), new StreamResult(new FileOutputStream(outOFxslFileName)));
    	}
	
	
	// se supone que este método será llamado al principio y acorde con la situación real
	public void definicionesTramos() throws JessException, java.io.IOException
	{System.out.println("definiciones tramos");
	double[] longitudes = {1, 260, 200, 290, 490, 290, 250, 290, 200, 250, 740, 402, 402, 350, 350, 150, 150, 390, 390, 350, 350, 200, 20, 20, 30};
	 int numeroTramos = longitudes.length;
	 int[][] conexiones = new int[numeroTramos][numeroTramos];
	 for(int i = 0; i < numeroTramos; i++)
	 {for(int j = 0; j < numeroTramos; j++)
		{conexiones[i][j]=0;
		}
	 }
	 conexiones[1][2]=1;
	 conexiones[1][22]=1;
	 conexiones[22][4]=1;
	 conexiones[4][5]=1;
	 conexiones[4][9]=1;
	 conexiones[5][6]=1;
	 conexiones[6][7]=1;
	 conexiones[9][8]=1;
	 conexiones[7][8]=1;
	 conexiones[8][10]=1;
	 conexiones[2][23]=1;
	 conexiones[23][21]=1;
	 conexiones[21][3]=1;
	 conexiones[21][4]=1;
	 conexiones[10][12]=1;
	  conexiones[1][24]=1;
	 conexiones[11][21]=1;
 conexiones[10][21]=1;
	 conexiones[12][13]=1;
	 conexiones[13][14]=1;
	 conexiones[12][15]=1;
	 conexiones[14][15]=1;
	 conexiones[14][11]=1;
	 conexiones[15][19]=1;
     conexiones[15][17]=1;
	 conexiones[17][18]=1;
	 conexiones[19][20]=1;
	 conexiones[20][16]=1;
	 conexiones[18][12]=1;
	 conexiones[16][13]=1;
	 conexiones[16][11]=1; 
	  conexiones[16][22]=1; 
	 String[] nombreTramos = new String[numeroTramos];
	 for (int j= 0; j< numeroTramos; j++)
	 {nombreTramos[j] = "Tramo" + (j);
	 }	 
		Vector vector = procesaTramos(longitudes, conexiones, nombreTramos);
	Vector vectorPrioridades = new Vector();
		Prioridades p1 = new Prioridades("Tramo21", "Tramo25");
		Prioridades p2 = new Prioridades("Tramo9", "Tramo7");
		Prioridades p3 = new Prioridades("Tramo10", "Tramo11");
		Prioridades p4 = new Prioridades("Tramo10", "Tramo23");
		Prioridades p5 = new Prioridades("Tramo10", "Tramo23");
	Prioridades p6 = new Prioridades("Tramo16", "Tramo14");
	Prioridades p7 = new Prioridades("Tramo21", "Tramo24");
	Prioridades p8 = new Prioridades("Tramo21", "Tramo22");
	
		vectorPrioridades.addElement(p1);
		vectorPrioridades.addElement(p2);
		vectorPrioridades.addElement(p3);
		vectorPrioridades.addElement(p4);
		vectorPrioridades.addElement(p5);
		vectorPrioridades.addElement(p6);	
		vectorPrioridades.addElement(p7);
		vectorPrioridades.addElement(p8);
	Vector vectorOposiciones = new Vector();
p1 = new Prioridades("Tramo12", "Tramo11");
		 p2 = new Prioridades("Tramo12", "Tramo16");
		 p3 = new Prioridades("Tramo16", "Tramo15");
		 p4 = new Prioridades("Tramo16", "Tramo12");
		p5 = new Prioridades("Tramo15", "Tramo19");
		p6 = new Prioridades("Tramo15", "Tramo20");
		p7 = new Prioridades("Tramo15", "Tramo18");
		p8 = new Prioridades("Tramo15", "Tramo17");
		vectorOposiciones.addElement(p1);
		vectorOposiciones.addElement(p2);
		vectorOposiciones.addElement(p3);
		vectorOposiciones.addElement(p4);
		vectorOposiciones.addElement(p5);
		vectorOposiciones.addElement(p6);
		vectorOposiciones.addElement(p7);
		vectorOposiciones.addElement(p8);
		meterVectorEnOntologia(vector, vectorPrioridades, vectorOposiciones);
	}
	
	public void calculaRuta() throws JessException, java.io.IOException
	   {calculaRuta("Tramo6", "Tramo1");
	   } 
	
	
	public Vector calculaRuta(String origen, String destino) throws JessException, java.io.IOException
	{String origenP = prefijo + origen;
	 String destinoP = prefijo + destino;
	 System.out.println("calcula ruta");	
	String comandoleer = "(compDist Final" + origenP + " Principio" + destinoP + ")";
		engineRutas.executeCommand(comandoleer);
		engineRutas.executeCommand("(batch consultasVerdino.clp)");
	    Value ruta = engineRutas.fetch("RUTA");
		Vector tramosRuta = new Vector();
		try {
		ValueVector v = ruta.listValue(engineRutas.getGlobalContext());
		
		String principio1 = "Principio";
			String finales1 = "Final";
		// la primera
		String nodo1 = v.get(0).stringValue(engineRutas.getGlobalContext());
		int indice1 = nodo1.indexOf(finales1);
		nodo1 = nodo1.substring(indice1 + finales1.length(),nodo1.length()).trim();
		tramosRuta.addElement(nodo1);
		//System.out.println(nodo1);
         for (int k=0; k<v.size(); k++)
		 {nodo1 = v.get(k).stringValue(engineRutas.getGlobalContext());
		  indice1 = nodo1.indexOf(principio1);
		  if(indice1 > -1)
		  {nodo1 = nodo1.substring(indice1 + principio1.length(),nodo1.length()).trim();
		   //System.out.println(nodo1);
		   tramosRuta.addElement(nodo1);
		  }
		 }
		 } catch (java.lang.NullPointerException e) {}
   	    return tramosRuta;
	}
	
	public  Vector procesaTramos (double[] longitudes, int[][] conexiones, String[] nombreTramos) throws JessException
	{System.out.println("procesa tramos");
	 Vector vector = new Vector();
	  int numeroTramos = longitudes.length; //el 0 es ficticio
	 for (int j= 0; j< numeroTramos; j++)
	 {String id = nombreTramos[j];
	  Tramo tramoj = new Tramo(id);
	  tramoj.fijaLongitud(longitudes[j]);
	  tramoj.fijaSucesores(conexiones[j]);
	  vector.add(tramoj);
	  }
	  return vector;
	}
	
	
	public void assertsTramos (Vector vector) throws JessException
	{int longitudVector = vector.size();
	 for (int i = 0; i<longitudVector; i++)
	 {Tramo tramoj = (Tramo) vector.elementAt(i);
	  double longitud = tramoj.dimeLongitud();
	  String id = tramoj.dimeId();
	  String assertsPropios = "(assert (Dist (t1 Principio" + id + ") (t2 Final" + id + ") (miles " + longitud + ")))";
	  engineRutas.executeCommand(assertsPropios);
      
	  Vector conexiones = tramoj.dimeVectorNombreSucesores();
	  for (int j = 0; j < conexiones.size(); j++)
	  {String idconectado = (String)conexiones.elementAt(j);
	    String assertsCruzados = "(assert (Dist (t1 Final" + id + ") (t2 Principio" + idconectado + ") (miles 0.0000000000001)))";
		engineRutas.executeCommand(assertsCruzados);
	   }
	 }
	}
	
	
	
public void meterVectorEnOntologia (Vector vector, Vector prioridades, Vector oposiciones) throws java.io.IOException
	{ System.out.println("meter vector en ontologia");
		String propiedadLongitud = prefijo + "tieneLongitud";
     // crear los tramos como instancias de Tramo
	   int size = vector.size();
	   for(int i=0; i<size; i++)
	   {Tramo tramo = (Tramo) vector.elementAt(i);
	    String nombre = tramo.dimeId();
	    Resource clase = m.getResource(prefijo + "Tramo");
	    Individual individuo = m.createIndividual(prefijo + nombre, clase);
		double longitud = tramo.dimeLongitud();
		DatatypeProperty propiedad = m.getDatatypeProperty(propiedadLongitud);
		individuo.addProperty(propiedad, Double.toString(longitud));
		ObjectProperty sucesor = m.getObjectProperty(prefijo + "tieneSucesor");
		//lo siguiente ralentiza
		int[] sucesores = tramo.dimeSucesores();
		for (int j = 0; j < sucesores.length; j++)
		{if (sucesores[j] > 0)
		 {Tramo tramoSucesor = (Tramo) vector.elementAt(j);
		  individuo.addProperty(sucesor, m.getResource(prefijo + tramoSucesor.dimeId()));
		 }
		}
	   }
	   String propiedadTP = "tieneTramoPrioritario";
	   String propiedadTS = "tieneTramoSecundario";
// escribir las prioridades en la ontología	   
	   for (int i= 0; i < prioridades.size(); i++)
	   {Prioridades prioridad = (Prioridades)prioridades.elementAt(i);
	    String nombre = "Prioridad" + i;
		Resource clase = m.getResource(prefijo + "InterseccionPrioritaria");
		Individual individuo = m.createIndividual(prefijo + nombre, clase);
		ObjectProperty tramoPrioritario = m.getObjectProperty(prefijo + propiedadTP);
		individuo.addProperty(tramoPrioritario, m.getResource(prefijo + prioridad.dimePrioritario()));
		ObjectProperty tramoSecundario = m.getObjectProperty(prefijo + propiedadTS);
		individuo.addProperty(tramoSecundario, m.getResource(prefijo + prioridad.dimeSecundario()));
	   }
	     for (int i= 0; i < oposiciones.size(); i++)
	   {Prioridades prioridad = (Prioridades)oposiciones.elementAt(i);
	    String nombre = "Oposicion" + i;
		Resource clase = m.getResource(prefijo + "Oposicion");
		Individual individuo = m.createIndividual(prefijo + nombre, clase);
		//System.out.println(individuo);
		ObjectProperty tramoPrioritario = m.getObjectProperty(prefijo + propiedadTP);
		individuo.addProperty(tramoPrioritario, m.getResource(prefijo + prioridad.dimePrioritario()));
		ObjectProperty tramoSecundario = m.getObjectProperty(prefijo + propiedadTS);
		individuo.addProperty(tramoSecundario, m.getResource(prefijo + prioridad.dimeSecundario()));
	   }
	   Writer out= new FileWriter("destino.owl");
	   m.write(out);
	}


public Vector dimeSucesoresTramo(String tramo) throws JessException
	 {//System.out.println("dime sucesdores tramos");
	 Vector sucesores = new Vector();
	  Individual individuo = m.getIndividual(tramo);
	  ObjectProperty tieneSucesores = m.getObjectProperty(prefijo + "tieneSucesor");
	  NodeIterator it = individuo.listPropertyValues(tieneSucesores);
	  while(it.hasNext())
	  {String nombre = (String) it.next().toString();
	   sucesores.add(nombre);
	  }
	  return sucesores;
	 }
	
	
	public void finalizaInicializacion() throws JessException
{System.out.println(System.currentTimeMillis());  
for (int u=0; u<70; u++)
	{engine.run();
	}
System.out.println(System.currentTimeMillis());
	}
	
		public void finalizaInicializacionReducida() throws JessException
{System.out.println("R" + System.currentTimeMillis());  
for (int u=0; u<10; u++)
	{engine.run();
	}
System.out.println("R" + System.currentTimeMillis());
	}
		
public String[] dimeEstados (String[] idVehiculos, String[] tramos, double[] longitudesEnTramos, double[] velocidades) throws JessException
{   Vector vectorVehiculos = new Vector();
	 for (Enumeration e = hashVehiculos.elements() ; e.hasMoreElements() ;) 
	{vectorVehiculos.addElement(e.nextElement());
     }
	 int sizeVector = vectorVehiculos.size();
	 System.out.println(":::: Hay " + sizeVector + " vehiculos !!!" );
	 
	 	engine.executeCommand(assertLimpiezaVelocidades);
	   engine.executeCommand(assertLimpiezaPosiciones);
	engine.run();
	engine.executeCommand(assertLimpiezaEstados);
	engine.run();
	for (int i=0; i<idVehiculos.length; i++)
	 {String idActual = prefijo + idVehiculos[i];
	  Vehiculo actual = (Vehiculo) (hashVehiculos.get(idActual));
	  //System.out.println(idActual);
	  double nuevaPosicion = longitudesEnTramos[i];
	      String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	 stringAssert = stringAssert + idActual + ") (object " + velocidades[i] +") ) ) ";
	  engine.executeCommand(stringAssert);
	  stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud) (subject ";
	 stringAssert = stringAssert + actual.dimePosicionVehiculo() + ") (object " + longitudesEnTramos[i] + ") ) ) ";
	 //System.out.println(actual.dimePosicionVehiculo());
	  engine.executeCommand(stringAssert);
      stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo) (subject ";
	 stringAssert = stringAssert + actual.dimePosicionVehiculo() + ") (object " + prefijo + tramos[i] + ") ) )";
	  engine.executeCommand(stringAssert);	
	 } 
	engine.run();
	String[] estados = new String[tramos.length];
	for (int i=0; i<idVehiculos.length ; i++)
	{estados[i]="Normal";
	 	String identificador = prefijo + idVehiculos[i];
	 // assert posicion
	 ValueVector vector2 = new ValueVector();
		vector2.add(new Value("http://www.isaatc.ull.es/Verdino.owl#tieneEstado", RU.ATOM));
		vector2.add(new Value(identificador,RU.ATOM));
        Iterator result2 =
            engine.runQuery("buscaObjetosConSujeto", vector2);
        String estado = " ";
		while (result2.hasNext()) {
			 Token t = (Token) result2.next();
             Fact f = (Fact) t.fact(1);
			estado = f.getSlotValue("object").stringValue(engine.getGlobalContext());
			
			
			if(!(estado.equals("http://www.isaatc.ull.es/Verdino.owl#Normal")))
			{//actual.fijaEstado("EsperaInterseccionPrioritaria");
			 estados[i]=quitaPrefijo(estado);
			}
		}	  
	}
		// quitar los hechos de espera
engine.executeCommand(assertLimpiezaTramos);
	engine.run();
	
	
 return estados;
}
		
	public void funcionamiento() throws JessException, java.io.IOException
	{System.out.println("funcionamiento");
	 Vector vectorVehiculos = new Vector();
	 for (Enumeration e = hashVehiculos.elements() ; e.hasMoreElements() ;) 
	{vectorVehiculos.addElement(e.nextElement());
     }
	// visualización
	Visualizacion visualizacion = new Visualizacion();
	 double[] longitudes = {1, 260, 200, 290, 490, 290, 250, 290, 200, 250, 740, 402, 402, 150, 150, 150, 150, 390, 390, 350, 350, 200, 20, 20, 30};
	 int numeroTramos = longitudes.length;
	String[] nombreTramos = new String[numeroTramos];
	 for (int j= 0; j< numeroTramos; j++)
	 {nombreTramos[j] = prefijo + "Tramo" + (j);
	 }	 
	 visualizacion.fijaTramos(nombreTramos);
	 visualizacion.fijaLongitudes(longitudes);
	 	 visualizacion.fijaVehiculos(hashVehiculos);
	int sizeVector = vectorVehiculos.size();
  System.out.println(":::: Hay " + sizeVector + " vehiculos !!!" );
  int instante = 1;
  //fin de inicialización
for (int u=0; u<70; u++)
{engine.run();
}
  // por cada segundo...
  while(true)
  {waiting(1000);
    System.out.println("--------Instante: " + instante + "----------");
    // mostrar la posición actual
	for (int i=0; i<sizeVector ; i++)
	{Vehiculo actual = (Vehiculo) (vectorVehiculos.elementAt(i));
		System.out.println(":::::");
		System.out.println(actual.dimeId());
		System.out.println("   velocidad= " + actual.dimeVelocidad());
		System.out.println("   " + actual.dimeTramo());
		System.out.println("   posicionTramo= " + actual.dimeLongitudEnTramo());
		System.out.println("   longitudTramo= " + dimeLongitudTramo(actual.dimeTramo()));
    }
	instante++;
	// calcular las posiciones nuevas 
	// NO HACERLO CON FUNCIONAMIENTO EN GPS
		System.out.println(engine.executeCommand(assertLimpiezaVelocidades));
	System.out.println(engine.executeCommand(assertLimpiezaPosiciones));
	engine.run();
	System.out.println(engine.executeCommand(assertLimpiezaEstados));
	engine.run();
	for (int i=0; i<sizeVector ; i++)
	{
	Vehiculo actual = (Vehiculo) (vectorVehiculos.elementAt(i));
		// caso de estar en el mismo tramo
		double nuevaPosicion = actual.dimeLongitudEnTramo() + actual.dimeVelocidad();
		double longitudTramo = dimeLongitudTramo(actual.dimeTramo());
		String tramo = actual.dimeTramo();
		if (!(actual.dimeEstado().equals("EnEspera")))
		{if (nuevaPosicion < longitudTramo)
		{actual.fijaLongitudEnTramo(nuevaPosicion);
		 visualizacion.recibeInformacion(actual.dimeId(), actual.dimeTramo(), actual.dimeLongitudEnTramo());
		    String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	 stringAssert = stringAssert + actual.dimeId() + ") (object " + actual.dimeVelocidad() +") ) ) ";
	  engine.executeCommand(stringAssert);
		 }
		else
		{String nuevoTramo = actual.dimeSiguienteTramo();
		 if (!(nuevoTramo.equals(" ")))
		 {actual.fijaTramo(nuevoTramo);
		  actual.fijaLongitudEnTramo(0);
		  visualizacion.recibeInformacion(actual.dimeId(), actual.dimeTramo(), actual.dimeLongitudEnTramo());
		    String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	 stringAssert = stringAssert + actual.dimeId() + ") (object " + actual.dimeVelocidad() +") ) ) ";
	  engine.executeCommand(stringAssert); 
	}
		 else
		 {actual.fijaVelocidad(0);
		    String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	 stringAssert = stringAssert + actual.dimeId() + ") (object 0) ) ) ";
	  engine.executeCommand(stringAssert);
		 }
		} // fin del else  
		}
		else
		{actual.fijaLongitudEnTramo(nuevaPosicion);
		 visualizacion.recibeInformacion(actual.dimeId(), actual.dimeTramo(), actual.dimeLongitudEnTramo());
		    String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	 stringAssert = stringAssert + actual.dimeId() + ") (object " + actual.dimeVelocidad() +") ) ) ";
	  engine.executeCommand(stringAssert);
		}
    }  // fin del for de cálculo de nuevas posiciones
	for (int i=0; i<sizeVector ; i++)
	{Vehiculo actual = (Vehiculo) (vectorVehiculos.elementAt(i));
	 // assert posicion
	  String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnLongitud) (subject ";
	 stringAssert = stringAssert + actual.dimePosicionVehiculo() + ") (object " + actual.dimeLongitudEnTramo() + ") ) ) ";
	  engine.executeCommand(stringAssert);
      stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#estaEnTramo) (subject ";
	 stringAssert = stringAssert + actual.dimePosicionVehiculo() + ") (object " + actual.dimeTramo() + ") ) )";
	  engine.executeCommand(stringAssert);	
	}
	engine.run();
	// comprobar si hay vehiculos en estado de espera
	for (int i=0; i<sizeVector ; i++)
	{Vehiculo actual = (Vehiculo) (vectorVehiculos.elementAt(i));
	 actual.fijaEstado("Normal");
   double velocidad = actual.dimeVelocidad();
	String identificador = actual.dimeId();
	 // assert posicion
	 ValueVector vector2 = new ValueVector();
		vector2.add(new Value("http://www.isaatc.ull.es/Verdino.owl#tieneEstado", RU.ATOM));
		vector2.add(new Value(identificador,RU.ATOM));
        Iterator result2 =
            engine.runQuery("buscaObjetosConSujeto", vector2);
        String estado = " ";
		while (result2.hasNext()) {
			 Token t = (Token) result2.next();
             Fact f = (Fact) t.fact(1);
			estado = f.getSlotValue("object").stringValue(engine.getGlobalContext());
			if(estado.equals("http://www.isaatc.ull.es/Verdino.owl#EnEspera"))
			{actual.fijaEstado("EnEspera");
			  String stringAssert = "(assert (triple (predicate http://www.isaatc.ull.es/Verdino.owl#tieneVelocidad) (subject ";
	stringAssert = stringAssert + actual.dimeId() + ") (object 0) ) )";
	  engine.executeCommand(stringAssert);	
			}
		}	  
	}
		// quitar los hechos de espera
	System.out.println(engine.executeCommand(assertLimpiezaTramos));
	engine.run();
  }  // fin del while
	}	
	
	public static void waiting (int n){
        long t0, t1;
        t0 =  System.currentTimeMillis();
     do{t1 = System.currentTimeMillis();
        } while (t1 - t0 < n);
    }

	
}

import java.util.Vector;

import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;

import jess.JessException;

/**
 * Clase se comunicación con el sistema inteligente de gestión de flota.
 * 
 * @author evelio
 */
public class InterfazFlota {

public Distancias distancias;

/**
 * Constructor vació que carga la ontología.
 * @throws JessException
 * @throws java.io.IOException
 * @throws TransformerException
 * @throws TransformerConfigurationException
 */
public InterfazFlota() throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{distancias = new Distancias();
distancias.cargaEspaciosDeNombres();
		//distancias.definicionesTramos();
		//distancias.transformaHechosJess();
		//distancias.transformaReglasJess();
		//distancias.arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		//distancias.cargarHechosReglas();
}

/**
 * Método en que se le indican al sistema los tramos, interconexiones y prioridades
 * @param tramos contendrá los nombres de los tramos
 * @param longitudes contendrá la longitud (en metros) de cada uno de los tramos
 * @param conexiones para indicar las conexiones. Deberá tener tantas filas y columnas como tramos. 
 * Cada fila representa el tramo inicial y cada columna el tramo siguiente. 
 * Por ejemplo si hay un 1 en conexiones[7][3] indica el que tramo 3 está a continuación del 7 
 * @param vectorPrioridades vector con las {@link Prioridades} en cruces de un tramo respecto a otro
 * @param vectorOposiciones vector con las {@link Prioridades} en oposición de un tramo respecto a otro
 * @throws JessException
 * @throws java.io.IOException
 * @throws TransformerException
 * @throws TransformerConfigurationException
 */
public void inicializacionTramos (String[] tramos, double[] longitudes, int[][] conexiones
		, Vector<Prioridades> vectorPrioridades, Vector<Prioridades> vectorOposiciones) 
throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{Vector vector = distancias.procesaTramos(longitudes, conexiones, tramos);
 	distancias.meterVectorEnOntologia(vector, vectorPrioridades, vectorOposiciones);
	distancias.transformaHechosJess();
		distancias.transformaReglasJess();
		distancias.arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		distancias.cargarHechosReglas();
	distancias.leerTramosDeOntologia();  
}

// si al final no se cogen de la ontolog�a...
/**
 * Indica los vehículos que están presentes
 * @param idVehiculos array con los nombres de los vehículos 
 */
public void inicializacionVehiculos(String[] idVehiculos) throws JessException
{distancias.inicializaVehiculos(idVehiculos);
}

/**
 * NO SE PARA QUE ES
 * @param idVehiculos
 * @param rutas
 * @throws JessException
 */
public void asignaRutasVehiculos(String[] idVehiculos, String[][] rutas) throws JessException
{distancias.limpiaRutas();
 for (int i=0; i< idVehiculos.length; i++)
 {asignaRuta(idVehiculos[i], rutas[i]);
 }
 distancias.finalizaInicializacion();
}

/**
 * NO SE PARA QUE ES
 * @param vehiculo
 * @param ruta
 * @throws JessException
 */
//no usar de manera aislada
public void asignaRuta(String vehiculo, String[] ruta) throws JessException
{distancias.asignaRuta(vehiculo, ruta);
 //distancias.finalizaInicializacionReducida();
}

/** 
 * Se pasa la posición y velocidad de todos los vehículo y devuelve en que estado deben ponerse.
 * @param idVehiculos identificación de los vehículos sobre los que se pregunta
 * @param tramosActuales tramos en que se encuentra el vehículo correspondiente
 * @param longitudesEnTramos posición en el tramo en que se encuetra el vehículo correspondiente
 * @param velocidades velocidad del vehículo correspondiente
 * @return array con el estado en que debe ponerse el vehículo correspondiente
 * @throws JessException
 */
public String[] dimeEstados (String[] idVehiculos, String[] tramosActuales, double[] longitudesEnTramos, double[] velocidades) throws JessException
{return distancias.dimeEstados(idVehiculos, tramosActuales, longitudesEnTramos, velocidades);
}

/**
 * Calcula la ruta más corta para ir de origen a destino
 * @param origen nombre del tramo origen
 * @param destino nombre del tramo destino
 * @return array con la suceción de tramos que se debe seguir (icluyendo origen y destino??)
 * @throws JessException
 * @throws java.io.IOException
 */
public String[] calculaRuta(String origen, String destino) throws JessException, java.io.IOException
{String[] strings = new String[0];
 if(origen.equals(destino))
 {strings = new String[1];
  strings[0]=destino;
  System.out.println(strings[0]);
 }
 else
 {
Vector vector = distancias.calculaRuta(origen, destino);
  strings = new String[vector.size()];
 for (int i=0; i< vector.size(); i++)
 {strings[i] = distancias.quitaPrefijo((String)vector.elementAt(i));
  System.out.println(strings[i]);
 }
 }
 return strings; 
}

public static void main(String[] args) throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{InterfazFlota interfaz = new InterfazFlota();
System.out.println("definiciones tramos");
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
		
	Vector<Prioridades> vectorPrioridades = new Vector<Prioridades>();
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
	Vector<Prioridades> vectorOposiciones = new Vector<Prioridades>();
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
	

	 interfaz.inicializacionTramos (nombreTramos, longitudes, conexiones, vectorPrioridades, vectorOposiciones);
     interfaz.calculaRuta("Tramo4", "Tramo3");
	 
	 String[] vehiculos = {"Verdino22", "VerdinoEspecial"};
	String[][] rutas = {{"Tramo10", "Tramo12"}, {"Tramo11", "Tramo21"}};
	interfaz.inicializacionVehiculos(vehiculos);
	interfaz.asignaRutasVehiculos(vehiculos,rutas);

	
	 String[] tramosActuales = {"Tramo10", "Tramo11"}; 
	 double[] longitudesEnTramos= {720, 240};
	 double[] velocidades = {10,10};
	String[] estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i]);
	 }
	 
	String[][] rutas2 = {{"Tramo10", "Tramo21"}, {"Tramo11", "Tramo21"}};
	interfaz.asignaRutasVehiculos(vehiculos,rutas2);
estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i]);
	 }
	 

	interfaz.asignaRutasVehiculos(vehiculos,rutas);
estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i]);
	 }
	 

String[] tramosActuales2 = {"Tramo4", "Tramo4"}; 
String[][] rutas3 = {{"Tramo4", "Tramo5"}, {"Tramo4", "Tramo5"}};
	 double[] longitudesEnTramos3= {200, 160};
	interfaz.asignaRutasVehiculos(vehiculos,rutas3);
estados = interfaz.dimeEstados(vehiculos, tramosActuales2, longitudesEnTramos3, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i]);
	 }
	 
	 String[] tramosActuales4 = {"Tramo7", "Tramo9"}; 
String[][] rutas4 = {{"Tramo7", "Tramo8"}, {"Tramo9", "Tramo8"}};
	 double[] longitudesEnTramos4= {245, 245};
	interfaz.asignaRutasVehiculos(vehiculos,rutas4);
estados = interfaz.dimeEstados(vehiculos, tramosActuales4, longitudesEnTramos4, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i]);
	 }
	 

	 }
}
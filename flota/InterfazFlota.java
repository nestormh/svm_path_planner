import java.util.*;
import jess.*;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

public class InterfazFlota {

public Distancias distancias;


public InterfazFlota() throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{distancias = new Distancias();
distancias.cargaEspaciosDeNombres();
		//distancias.definicionesTramos();
		//distancias.transformaHechosJess();
		//distancias.transformaReglasJess();
		//distancias.arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		//distancias.cargarHechosReglas();
}

public void inicializacionTramos (String[] tramos, double[] longitudes, int[][] conexiones, Vector vectorPrioridades, Vector vectorOposiciones) throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{Vector vector = distancias.procesaTramos(longitudes, conexiones, tramos);
 	distancias.meterVectorEnOntologia(vector, vectorPrioridades, vectorOposiciones);
	distancias.transformaHechosJess();
		distancias.transformaReglasJess();
		distancias.arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		distancias.cargarHechosReglas();
	distancias.leerTramosDeOntologia();  
}

// si al final no se cogen de la ontología...
public void inicializacionVehiculos(String[] idVehiculos) throws JessException
{distancias.inicializaVehiculos(idVehiculos);
}

public void asignaRutasVehiculos(String[] idVehiculos, String[][] rutas) throws JessException
{distancias.limpiaRutas();
 for (int i=0; i< idVehiculos.length; i++)
 {asignaRuta(idVehiculos[i], rutas[i]);
 }
 distancias.finalizaInicializacion();
}

//no usar de manera aislada
public void asignaRuta(String vehiculo, String[] ruta) throws JessException
{distancias.asignaRuta(vehiculo, ruta);
 //distancias.finalizaInicializacionReducida();
}

public String[] dimeEstados (String[] idVehiculos, String[] tramosActuales, double[] longitudesEnTramos, double[] velocidades) throws JessException
{return distancias.dimeEstados(idVehiculos, tramosActuales, longitudesEnTramos, velocidades);
}


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
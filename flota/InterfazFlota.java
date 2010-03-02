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


public InterfazFlota() 
{
try{
distancias = new Distancias();
distancias.cargaEspaciosDeNombres();
} catch (Exception e) {e.printStackTrace();}
		
}

public void inicializacionTramos (String[] tramos, double[] longitudes, int[][] conexiones, Vector vectorPrioridades, Vector vectorOposiciones) 
{try {
 Vector vector = distancias.procesaTramos(longitudes, conexiones, tramos);
 	distancias.meterVectorEnOntologia(vector, vectorPrioridades, vectorOposiciones);
	distancias.transformaHechosJess();
		distancias.transformaReglasJess();
		distancias.arreglarReglas("reglasverdino.clp", "reglasarregladas.clp");
		distancias.cargarHechosReglas();
	distancias.leerTramosDeOntologia();  
	} catch (Exception e) {e.printStackTrace();}
}

// si al final no se cogen de la ontología...
public void inicializacionVehiculos(String[] idVehiculos) 
{try{
 distancias.inicializaVehiculos(idVehiculos);
 } catch (Exception e) {e.printStackTrace();}
}

public void asignaRutasVehiculos(String[] idVehiculos, String[][] rutas) 
{try {
 distancias.limpiaRutas();
 for (int i=0; i< idVehiculos.length; i++)
 {asignaRuta(idVehiculos[i], rutas[i]);
 }
 distancias.finalizaInicializacion();
 } catch (Exception e) {e.printStackTrace();}
}

//no usar de manera aislada
public void asignaRuta(String vehiculo, String[] ruta) 
{try{
 distancias.asignaRuta(vehiculo, ruta);
 } catch (Exception e) {e.printStackTrace();}
 //distancias.finalizaInicializacionReducida();
}

public Conflicto[] dimeEstados (String[] idVehiculos, String[] tramosActuales, double[] longitudesEnTramos, double[] velocidades) 
{try{
 return distancias.dimeEstados(idVehiculos, tramosActuales, longitudesEnTramos, velocidades);
 }catch (Exception e) {e.printStackTrace(); return null;}
 }


public String[] calculaRuta(String origen, String destino) 
{String[] strings = new String[0];
 try {
 //if(origen.equals(destino))
// {strings = new String[1];
 // strings[0]=destino;
 // System.out.println(strings[0]);
 //}
 //else
 //{
Vector vector = distancias.calculaRuta(origen, destino);
  strings = new String[vector.size()];
 for (int i=0; i< vector.size(); i++)
 {strings[i] = distancias.quitaPrefijo((String)vector.elementAt(i));
  //System.out.println(strings[i]);
 }
 //}
 } catch (Exception e) {e.printStackTrace();}
 return strings; 
}

public static void main(String[] args) throws JessException, java.io.IOException, TransformerException, TransformerConfigurationException
{InterfazFlota interfaz = new InterfazFlota();

// longitudes en unidades de medida de los tramos
// para no tener un tramo0, creo el primero extra, aunque no se usen
	double[] longitudes = {1, 260, 200, 290, 490, 290, 250, 290, 200, 250, 740, 402, 402, 350, 350, 150, 150, 390, 390, 350, 350, 200, 20, 20, 30};
	 int numeroTramos = longitudes.length;
// nombre de los tramos. pueden ser escogidos de forma arbitraria (en cualquier formato)
// aquí se han tomado como "Tramo1", "Tramo2", etc... pero no es necesario	 
	 String[] nombreTramos = new String[numeroTramos];
	 for (int j= 0; j< numeroTramos; j++)
	 {nombreTramos[j] = "Tramo" + (j);
	 }	 
// conexiones entre tramos a nivel topológico
// conexiones[predecesor][sucesor]
// si =1 hay conexión (en realidad distinto de 0)	 
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
	 
// prioridades de tipo tramo sobre tramo
// Prioridades ("TramoPrioritario", "TramoSecundario");
// se deben meter en un vector		
	Vector vectorPrioridades = new Vector();
		vectorPrioridades.addElement(new Prioridades("Tramo21", "Tramo25"));
		vectorPrioridades.addElement(new Prioridades("Tramo9", "Tramo7"));
		vectorPrioridades.addElement(new Prioridades("Tramo10", "Tramo11"));
		vectorPrioridades.addElement(new Prioridades("Tramo10", "Tramo23"));
		vectorPrioridades.addElement(new Prioridades("Tramo10", "Tramo23"));
		vectorPrioridades.addElement(new Prioridades("Tramo16", "Tramo14"));	
		vectorPrioridades.addElement(new Prioridades("Tramo21", "Tramo24"));
		vectorPrioridades.addElement(new Prioridades("Tramo21", "Tramo22"));
		
// prioridades de tipo oposición sobre tramos
// son conmutativas, por lo que no hace falta definir la inversa
// Ej. no se puede invadir el Tramo12 si el Tramo11 está ocupado			
	Vector vectorOposiciones = new Vector();
		vectorOposiciones.addElement(new Prioridades("Tramo12", "Tramo11"));
		vectorOposiciones.addElement(new Prioridades("Tramo12", "Tramo16"));
		vectorOposiciones.addElement(new Prioridades("Tramo16", "Tramo15"));
		vectorOposiciones.addElement(new Prioridades("Tramo16", "Tramo12"));
		vectorOposiciones.addElement(new Prioridades("Tramo15", "Tramo19"));
		vectorOposiciones.addElement(new Prioridades("Tramo15", "Tramo20"));
		vectorOposiciones.addElement(new Prioridades("Tramo15", "Tramo18"));
		vectorOposiciones.addElement(new Prioridades("Tramo15", "Tramo17"));
	
// inicialización
	 interfaz.inicializacionTramos (nombreTramos, longitudes, conexiones, vectorPrioridades, vectorOposiciones);
// ejemplo de cálculo de ruta
// OJO. NO FUNCIONA EN EL CASO DE QUE EL DESTINO ESTÉ MÁS ADELANTE EN EL MISMO TRAMO. ES UN CASO PARTICULAR. 
	 String[] rutaCalculada = interfaz.calculaRuta("Tramo4", "Tramo4");
		for (int i=0; i< rutaCalculada.length; i++)
		{System.out.println(rutaCalculada[i]);
		}	

// inicialización vehículos y de sus rutas (por orden)
// ejemplo: Primero voy por el tramo 10, y después deseo ir al 12-
// Necesario para prioridades del tipo oposición (tengo que saber cuál es el próximo tramo a visitar 
// y ver si hay algún problema
	 String[] vehiculos = {"Verdino22", "VerdinoEspecial"};
	String[][] rutas = {{"Tramo10", "Tramo12"}, {"Tramo11", "Tramo21"}};
	interfaz.inicializacionVehiculos(vehiculos);
	interfaz.asignaRutasVehiculos(vehiculos,rutas);

// Ejemplo de detección de conflictos
// tipos EsperaDistancia (dos coches en el mismo tramo) EsperaInterseccionPrioritaria y EsperaOposicion
	 String[] tramosActuales = {"Tramo10", "Tramo11"}; 
	 double[] longitudesEnTramos= {720, 240};
	 double[] velocidades = {10,10};
	Conflicto[] estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i].dimeTipo() + " a " + estados[i].dimeDistancia() + " unidades");
	 }
	 
	String[][] rutas2 = {{"Tramo10", "Tramo21"}, {"Tramo11", "Tramo21"}};
	interfaz.asignaRutasVehiculos(vehiculos,rutas2);
estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i].dimeTipo() + " a " + estados[i].dimeDistancia() + " unidades");
	 }
	 

	interfaz.asignaRutasVehiculos(vehiculos,rutas);
estados = interfaz.dimeEstados(vehiculos, tramosActuales, longitudesEnTramos, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i].dimeTipo() + " a " + estados[i].dimeDistancia() + " unidades");
	 }
	 

String[] tramosActuales2 = {"Tramo4", "Tramo4"}; 
String[][] rutas3 = {{"Tramo4", "Tramo5"}, {"Tramo4", "Tramo5"}};
	 double[] longitudesEnTramos3= {200, 160};
	interfaz.asignaRutasVehiculos(vehiculos,rutas3);
estados = interfaz.dimeEstados(vehiculos, tramosActuales2, longitudesEnTramos3, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i].dimeTipo() + " a " + estados[i].dimeDistancia() + " unidades");
	 }
	 
	 String[] tramosActuales4 = {"Tramo7", "Tramo9"}; 
String[][] rutas4 = {{"Tramo7", "Tramo8"}, {"Tramo9", "Tramo8"}};
	 double[] longitudesEnTramos4= {245, 245};
	interfaz.asignaRutasVehiculos(vehiculos,rutas4);
estados = interfaz.dimeEstados(vehiculos, tramosActuales4, longitudesEnTramos4, velocidades);
	 for (int i=0; i< estados.length; i++)
	 {System.out.println(vehiculos[i] + "," + estados[i].dimeTipo() + " a " + estados[i].dimeDistancia() + " unidades");
	 }
	 

	 }
}
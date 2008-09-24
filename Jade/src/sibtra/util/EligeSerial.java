package sibtra.util;

import java.util.Enumeration;
import java.util.Vector;

import gnu.io.CommPortIdentifier;

import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JOptionPane;

/**
 * Abre ventana donde elegir seriales para distintos elementos.
 * Lista de seriales dispontibles la toma de RXTX.
 * Título de seriales a elegir se deben pasar en el constructor.
 * @author alberto
 *
 */
public class EligeSerial {

	private JFrame jfDialogo;

	private Vector<String> vPuertos;

	private int[] asignacion;

	private int numPuertos;
	
	public EligeSerial(String[] titulos) {
		asignacion=null;
		if(titulos==null) return;
		numPuertos=titulos.length;
		if(numPuertos==0) return;
		jfDialogo=new JFrame("Selección de seriales");
		Enumeration portList = CommPortIdentifier.getPortIdentifiers();
		
		//Pasamos lista de puertos a vector
		vPuertos=new Vector<String>();
		while (portList.hasMoreElements()) {
		    CommPortIdentifier portId = (CommPortIdentifier) portList.nextElement();

		    if (portId.getPortType() == CommPortIdentifier.PORT_SERIAL) {
		    	vPuertos.add(portId.getName());
			    System.out.println("Found port " + vPuertos.lastElement());
		    }
		}
		
		//vemos si hay suficientes seriales para los títulos pasados
		if(vPuertos.size()<numPuertos) {
			JOptionPane.showMessageDialog(jfDialogo,
			    "El número de puertos seriales es "+vPuertos.size()+" y se necesitan "+numPuertos,
			    "Error",
			    JOptionPane.ERROR_MESSAGE);
			return;
		}
		//hay suficientes puertos, abrimos la ventana para que los elija
		asignacion=new int[numPuertos];
		if(vPuertos.size()==1) {
			JOptionPane.showMessageDialog(jfDialogo,
				    "Sólo hay 1 puerto seriales ("+vPuertos.elementAt(0)+") y se asigna",
				    "Información",
				    JOptionPane.INFORMATION_MESSAGE);
			asignacion[0]=0;
			return;			
		}
	}
	
	public String[] getPuertos() {
		if(asignacion==null)
			return null;
		String[] puertos=new String[numPuertos];
		for(int i=0; i<numPuertos; i++)
			puertos[i]=vPuertos.elementAt(asignacion[i]);
		return puertos;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Apéndice de método generado automáticamente
		String[] titulos={"GPS"};//{"GPS","IMU","RF"};
		EligeSerial es=new EligeSerial(titulos);
		
		String[] puertos=es.getPuertos();
		System.out.println("Asignación de puertos:");
		for(int i=0;i<puertos.length;i++)
			System.out.println(i+"- "+puertos[i]);

	}

}

package sibtra.util;

import gnu.io.CommPortIdentifier;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Enumeration;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

/**
 * Abre ventana donde elegir seriales para distintos elementos.
 * Lista de seriales dispontibles la toma de RXTX.
 * Título de seriales a elegir se deben pasar en el constructor.
 * @author alberto
 *
 */
public class EligeSerial implements ActionListener {

	/** puertos encontrados por RXTX */
	private Vector<String> vPuertos;

	/** asingación de seriales echa por usuario, null si no es posible */
	private int[] asignacion=null;

	/** Para indicar a {@link #getPuertos()} que se ha hecho selección */
	private boolean hayAsignacion=false;

	/** Numero de puertos que se piden */
	private int numPuertos;

	/** Ventana principal */
	private JFrame jfPregunta;

	/** Combos de selección del puerto para cada título */
	private JComboBox[] jcbPuertos;

	/** Botón asignar*/
	private JButton jbAsignar;

	/** Botón de cancelar */
	private JButton jbCancelar;

	/** Refencia a thread que llama a {@link #getPuertos()} y que debe esperar {@link Thread.sleep()} 
	 * hasta que el usuario asigne.
	 */
	private Thread threadPrincipal=null;
	
	
	/**
	 * @param titulos Titulo de los subsistemas para los que se busca puerto serial
	 */ 
	public EligeSerial(String[] titulos) {
		if(titulos==null) { hayAsignacion=true; return; }
		numPuertos=titulos.length;
		if(numPuertos==0) { hayAsignacion=true; return; }
		jfPregunta=new JFrame("Selección de seriales");
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
			JOptionPane.showMessageDialog(jfPregunta,
			    "El número de puertos seriales es "+vPuertos.size()+" y se necesitan "+numPuertos,
			    "Error",
			    JOptionPane.ERROR_MESSAGE);
			hayAsignacion=true;
			return;
		}
		//hay suficientes puertos, abrimos la ventana para que los elija
		asignacion=new int[numPuertos];
		if(vPuertos.size()==1) {
			JOptionPane.showMessageDialog(jfPregunta,
				    "Sólo hay 1 puerto seriales ("+vPuertos.elementAt(0)+") y se asigna",
				    "Información",
				    JOptionPane.INFORMATION_MESSAGE);
			asignacion[0]=0;
			hayAsignacion=true;
			return;			
		}
		//hay más de 1 puerto abrimos la ventana para que elijan
		jcbPuertos=new JComboBox[numPuertos];
		{ //panel central para la elección
			JPanel jpCentral=new JPanel(new GridLayout(0,2));
			for(int i=0; i<numPuertos; i++) {
				jpCentral.add(new JLabel(titulos[i],SwingConstants.RIGHT));
			    jcbPuertos[i] = new JComboBox(vPuertos);
			    jcbPuertos[i].setSelectedIndex(i);
			    jpCentral.add(jcbPuertos[i]);
			}
			jfPregunta.add(jpCentral,BorderLayout.CENTER);
		}
		{  //Parte inferior para los botones
			JPanel jpSur=new JPanel();
			jbAsignar=new JButton("Asignar");
			jpSur.add(jbAsignar);
			jbAsignar.addActionListener(this);
			jbCancelar=new JButton("Cancelar");
			jpSur.add(jbCancelar);
			jbCancelar.addActionListener(this);
			
			jfPregunta.add(jpSur,BorderLayout.SOUTH);
			
		}
		jfPregunta.setDefaultCloseOperation(
			    JFrame.DO_NOTHING_ON_CLOSE);
		//cerrar la ventana es como pulsar el cancel
		jfPregunta.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent we) {
				jbCancelar.doClick();
			}
		});

		jfPregunta.pack();
		jfPregunta.setVisible(true);
	}
	
	/**
	 * Devuleve los nombres de los puertos seleccionados para cada uno de los título.
	 * Espera a que el usuario haga la selección o cancele.
	 * En caso de cancelar o que no sea posible la selección (puertos<títulos) devuelve NULL.
	 * Si sólo hay un puerto para 1 titulo se asigna sin preguntar al usuario
	 * @return
	 */
	public String[] getPuertos() {
		while(!hayAsignacion) {
			threadPrincipal=Thread.currentThread();
			try {
				Thread.sleep(10000);
			} catch (Exception e) {}
		}
		if(asignacion==null)
			return null;
		String[] puertos=new String[numPuertos];
		for(int i=0; i<numPuertos; i++)
			puertos[i]=vPuertos.elementAt(asignacion[i]);
		return puertos;
	}

	/**
	 * atiende las pulsaciones de los botones
	 */
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==jbCancelar) {
			jfPregunta.setVisible(false);
			asignacion=null; //no hay asignacion
			hayAsignacion=true;
			if(threadPrincipal!=null)
				threadPrincipal.interrupt(); //despertamos thread principal
			return;
		}
		if(e.getSource()==jbAsignar) {
			//copiamos asignaciones y comprobamos si hay repetidas
			boolean repetido=false;
			for(int i=0; !repetido && i<numPuertos;i++) {
				asignacion[i]=jcbPuertos[i].getSelectedIndex();
				for(int j=0; !repetido && j<i; j++) {
					repetido= (asignacion[i]==asignacion[j]);
				}
			}
			if(!repetido) {
				//todo correcto
				jfPregunta.dispose();
				hayAsignacion=true;
				if(threadPrincipal!=null)
					threadPrincipal.interrupt(); //despertamos thread principal
				return;
			}
			//advertimos de la repetición
			SwingUtilities.invokeLater(new Runnable(){
				public void run() {
					JOptionPane.showMessageDialog(jfPregunta,
							"Las seriales elegidas deben ser distintas",
							"Error seleccion",
							JOptionPane.ERROR_MESSAGE);

				}
			});
		}
	}

	/**
	 * Probamos la clase
	 */
	public static void main(String[] args) {
		//String[] titulos={"GPS"};
		String[] titulos={"GPS","IMU","RF"};
		
		String[] puertos=new EligeSerial(titulos).getPuertos();
		System.out.println("Asignación de puertos:");
		if(puertos==null) {
			System.out.println("No se asignaron puertos");
		} else
			for(int i=0;i<puertos.length;i++)
				System.out.println(i+"- "+puertos[i]);

	}


}

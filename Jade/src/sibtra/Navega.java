package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JCheckBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import sibtra.gps.GPSConnection;
import sibtra.gps.GPSData;
import sibtra.gps.GpsEvent;
import sibtra.gps.GpsEventListener;
import sibtra.gps.PanelMuestraGPSData;
import sibtra.gps.Ruta;
import sibtra.gps.SimulaGps;
import sibtra.imu.AngulosIMU;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelMuestraAngulosIMU;
import sibtra.lms.BarridoAngular;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculoSubjetivo;

/**
 * Para realizar la navegación (sin control del coche) detectando obstáculos con el RF.
 * @author alberto
 *
 */
public class Navega implements GpsEventListener {
	
	private ConexionSerialIMU csi;
	private GPSConnection gpsCon;
	private ManejaLMS manLMS;
	private JFrame ventGData;
	private PanelMuestraGPSData PMGPS;
	private JFrame ventIMU;
	private PanelMuestraAngulosIMU pmai;
	private JFileChooser fc;
	private Ruta rutaEspacial;
	private JLabel jlNomF;
	
	double[][] Tr=null;
	private MiraObstaculo mi;
	private JFrame ventanaPMOS;
	private JFrame ventanaPMO;
	private PanelMiraObstaculo pmo;
	private PanelMiraObstaculoSubjetivo PMOS;
	private double desMag;
	private JCheckBox jcbNavegando;

	public Navega(String[] args) {
		if(args.length<3) {
			System.err.println("Son necesarios 3 argumentos con los puerstos seriales");
			System.exit(1);
		}
		
		
		System.out.println("Abrimos conexión IMU");
		csi=new ConexionSerialIMU();
		if(!csi.ConectaPuerto(args[1],5)) {
			System.err.println("Problema en conexión serial con la IMU");
			System.exit(1);
		}
		
		//comunicación con GPS
		gpsCon=new SimulaGps(args[0]).getGps();
		if(gpsCon==null) {
			System.err.println("No se obtuvo GPSConnection");
			System.exit(1);
		}
		gpsCon.setCsIMU(csi);
		
		
		//Conectamos a RF
		try { 		
			manLMS=new ManejaLMS(args[2]);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF");
		}
		
		//VEntana datos gps
		ventGData=new JFrame("Datos GPS");
		PMGPS=new PanelMuestraGPSData(false);
		PMGPS.actualizaPunto(new GPSData()); 

		ventGData.getContentPane().add(PMGPS,BorderLayout.CENTER);
		ventGData.pack();
		ventGData.setVisible(true);
		gpsCon.addGpsEventListener(PMGPS);


		//Creamos ventana para IMU
		ventIMU=new JFrame("Datos IMU");
		pmai=new PanelMuestraAngulosIMU();
		pmai.actualizaAngulo(new AngulosIMU(0,0,0,0));
		ventIMU.add(pmai);

		ventIMU.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventIMU.pack();
		ventIMU.setVisible(true);
		//conecto manejador cuando todas las ventanas están creadas
		csi.addIMUEventListener(pmai);


		//elegir fichero
		fc=new JFileChooser(new File("./Rutas"));

		//necestamos leer archivo con la ruta
		boolean seCargo=false;
		do {
			int devuelto=fc.showOpenDialog(ventGData);
			if (devuelto!=JFileChooser.APPROVE_OPTION) 
				JOptionPane.showMessageDialog(ventGData,
						"Necesario cargar fichero de ruta",
						"Error",
						JOptionPane.ERROR_MESSAGE);
			else  {
				File file=fc.getSelectedFile();
				try {
					ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
					rutaEspacial=(Ruta)ois.readObject();
					ois.close();
					jlNomF.setText("Fichero: "+file.getName());
					seCargo=true;
				} catch (IOException ioe) {
					JOptionPane.showMessageDialog(ventGData,
							"Error al abrir el fichero " + file.getName(),
							"Error",
							JOptionPane.ERROR_MESSAGE);
					System.err.println(ioe.getMessage());
				} catch (ClassNotFoundException cnfe) {
					JOptionPane.showMessageDialog(ventGData,
							"Objeto leído inválido",
							"Error",
							JOptionPane.ERROR_MESSAGE);
					System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
				}
			}
		} while(!seCargo);
		
		desMag=rutaEspacial.getDesviacionM();
		System.out.println("Usando desviación magnética "+desMag);
		
		//Rellenamos la tryectoria
		Tr=new double[rutaEspacial.getNumPuntos()][2];
		for(int i=0; i<rutaEspacial.getNumPuntos();i++) {
			Tr[i][0]=rutaEspacial.getPunto(i).getXLocal();
			Tr[i][1]=rutaEspacial.getPunto(i).getYLocal();
		}
		
		
		System.out.println("Longitud de la trayectoria="+Tr.length);
		
		mi=new MiraObstaculo(Tr);
		try {
			PMOS=new PanelMiraObstaculoSubjetivo(mi,(short)manLMS.getDistanciaMaxima());
		} catch (LMSException e) {
			System.err.println("Problema al obtener distancia maxima configurada");
			System.exit(1);
		}
		
		//Checkbox para navegar
		jcbNavegando=new JCheckBox("Navegando");
		jcbNavegando.setSelected(false);
		ventanaPMOS.getContentPane().add(jcbNavegando,BorderLayout.SOUTH);
		
		ventanaPMOS=new JFrame("Mira Obstáculo Subjetivo");
		ventanaPMOS.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPMOS.getContentPane().add(PMOS,BorderLayout.CENTER);
		ventanaPMOS.setSize(new Dimension(800,400));
		ventanaPMOS.setVisible(true);

		
		ventanaPMO=new JFrame("Mira Obstáculo");
		ventanaPMO.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pmo=new PanelMiraObstaculo(mi);
		ventanaPMO.getContentPane().add(pmo,BorderLayout.CENTER);
		ventanaPMO.setSize(new Dimension(800,600));
		ventanaPMO.setVisible(true);
		
		//conecto manejador cuando todas las ventanas están creadas
		gpsCon.addGpsEventListener(this);
	}

	public void handleGpsEvent(GpsEvent ev) {
		if(ev.isEspacial() && jcbNavegando.isSelected()) {
			//Tenemos un nuevo punto ESPACIAL, pedimos barrido y miramos si hay obstáculo
			//Damos pto, orientación y barrido
			try {
				manLMS.pideBarrido((short)0, (short)180, (short)1);
				BarridoAngular ba=manLMS.recibeBarrido();
				double[] ptoAct={ev.getNuevoPunto().getXLocal(), ev.getNuevoPunto().getYLocal()};
				double dist=mi.masCercano(ptoAct, Math.toRadians(csi.getAngulo().getYaw())+desMag, ba);
				pmo.actualiza();
				PMOS.actualiza();
				if(Double.isInfinite(dist))
					System.out.println("Estamos fuera del camino");
				else
					System.out.println("Distancia="+dist);
					
//				System.out.println(" iAD="+PMOS.MI.iAD
//				+"\n iAI="+PMOS.MI.iAI
//				+"\n iptoD ="+PMOS.MI.iptoD
//				+" \n iptoI ="+PMOS.MI.iptoI
//				+" \n iptoDini ="+PMOS.MI.iptoDini
//				+" \n iptoIini ="+PMOS.MI.iptoIini
//				+" \n imin ="+PMOS.MI.indMin
//				);
			} catch (LMSException e) {
				System.err.println("Problemas al obtener barrido en punto "+ev.getNuevoPunto()
						+" :"+e.getMessage());
			}
		}
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Navega na=new Navega(args);
	}

}

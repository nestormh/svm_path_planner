package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.File;

import javax.swing.JCheckBox;
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
import sibtra.predictivo.Coche;
import sibtra.predictivo.ControlPredictivo;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculoSubjetivo;
import sibtra.util.EligeSerial;

/**
 * Para realizar la navegación (sin control del coche) detectando obstáculos con el RF.
 * @author alberto
 *
 */
public class NavegaPredictivo  {
	/** Milisegundos del ciclo */
	private static final long miliEspera = 200;
	
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
	JCheckBox jcbNavegando;
	
	Coche modCoche;
	ControlPredictivo cp;
	

	/** Se le han de pasar los 3 puertos series para: IMU, GPS, RF y Coche (en ese orden)*/
	public NavegaPredictivo(String[] args) {
		if(args==null || args.length<4) {
			System.err.println("Son necesarios 4 argumentos con los puertos seriales");
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
		do {
			int devuelto=fc.showOpenDialog(ventGData);
			if (devuelto!=JFileChooser.APPROVE_OPTION) 
				JOptionPane.showMessageDialog(ventGData,
						"Necesario cargar fichero de ruta",
						"Error",
						JOptionPane.ERROR_MESSAGE);
			else  {
				gpsCon.loadRuta(fc.getSelectedFile().getAbsolutePath());
			}
		} while(gpsCon.getRutaEspacial()==null);
		//nuestra ruta espacial será la que se cargó
		rutaEspacial=gpsCon.getRutaEspacial();
		desMag=rutaEspacial.getDesviacionM();
		System.out.println("Usando desviación magnética "+desMag);
		
		//Rellenamos la trayectoria
		Tr=rutaEspacial.toTr();
		
		
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
		
		ventanaPMOS=new JFrame("Mira Obstáculo Subjetivo");
		ventanaPMOS.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPMOS.getContentPane().add(PMOS,BorderLayout.CENTER);
		ventanaPMOS.getContentPane().add(jcbNavegando,BorderLayout.SOUTH);
		ventanaPMOS.setSize(new Dimension(800,400));
		ventanaPMOS.setVisible(true);

		
		ventanaPMO=new JFrame("Mira Obstáculo");
		ventanaPMO.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pmo=new PanelMiraObstaculo(mi);
		ventanaPMO.getContentPane().add(pmo,BorderLayout.CENTER);
		ventanaPMO.setSize(new Dimension(800,600));
		ventanaPMO.setVisible(true);
		
		//conecto manejador cuando todas las ventanas están creadas
//		gpsCon.addGpsEventListener(this);
	}


	/**
	 * Crea el objeto y queda en bucle en que se ejecuta cada {@link #miliEspera} un nuevo 
	 * cálculo de distancia del obstáculo.
	 * @param args Seriales para IMU, GPS y RF. Si no se pasan de piden interactivamente.
	 */
	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<3) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"IMU","GPS","RF","Coche"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
        Coche carroOri = new Coche();
        double vel = 2;
        double consVolante = 0;
        //TODO leer las cosas del coche
        carroOri.setVelocidad(vel);
        carroOri.setConsignaVolante(consVolante);

		
		NavegaPredictivo na=new NavegaPredictivo(puertos);
		
		long tSig;
		while(true) {
			tSig=System.currentTimeMillis()+miliEspera;
			try { 
				if(na.jcbNavegando.isSelected()) {
					na.manLMS.pideBarrido((short)0, (short)180, (short)1);
					BarridoAngular ba=na.manLMS.recibeBarrido();
					GPSData pa=na.gpsCon.getPuntoActualTemporal();
					double[] ptoAct={pa.getXLocal(), pa.getYLocal()};
					double dist=na.mi.masCercano(ptoAct
							, Math.toRadians(pa.getAngulosIMU().getYaw())+na.desMag, ba);
					na.pmo.actualiza();
					na.PMOS.actualiza();
					if(Double.isNaN(dist))
						System.out.println("Estamos fuera del camino");
					else if(Double.isInfinite(dist))
						System.out.println("No hay obstáculo");
					else
						System.out.println("Distancia="+dist);
				}
			} catch (LMSException e) {
				System.err.println("Problemas al obtener barrido en punto "
						+" :"+e.getMessage());
			}
			//esperamos hasta que hayan pasado miliSeg de ciclo.
			while(System.currentTimeMillis()<tSig)
				try{Thread.sleep(tSig-System.currentTimeMillis());} catch (Exception e) {}				
		}
	}

}

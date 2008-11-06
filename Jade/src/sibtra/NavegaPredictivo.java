package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.File;

import javax.swing.BoxLayout;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.PanelCoche;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.GPSData;
import sibtra.gps.GpsEvent;
import sibtra.gps.GpsEventListener;
import sibtra.gps.PanelMuestraGPSData;
import sibtra.gps.Ruta;
import sibtra.imu.AngulosIMU;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelMuestraAngulosIMU;
import sibtra.lms.BarridoAngular;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.predictivo.Coche;
import sibtra.predictivo.CocheModeloAntiguo;
import sibtra.predictivo.ControlPredictivo;
import sibtra.predictivo.PanelMuestraPredictivo;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculoSubjetivo;
import sibtra.util.EligeSerial;

/**
 * Para realizar la navegación controlando el coche con @link {@link ControlPredictivo}
 *  detectando obstáculos con el RF.
 * @author alberto
 *
 */
public class NavegaPredictivo implements GpsEventListener  {
	/** Milisegundos del ciclo */
	private static final long periodoMuestreoMili = 200;

	private static final double COTA_ANGULO = Math.toRadians(30);
	
	private ConexionSerialIMU csi;
	private GPSConnectionTriumph gpsCon;
	private ManejaLMS manLMS;
	private JFrame ventNumeros;
	private PanelMuestraGPSData PMGPS;
	private PanelMuestraAngulosIMU pmai;
	private JFileChooser fc;
	private Ruta rutaEspacial;
	
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
	ControlCarro contCarro;

	private PanelMuestraPredictivo pmp;

	private JLabel jlCalidad;

	private PanelCoche pmCoche;

	private JLabel jlNumPaquetes;

	/** Se le han de pasar los 3 puertos series para: IMU, GPS, RF y Coche (en ese orden)*/
	public NavegaPredictivo(String[] args) {
		if(args==null || args.length<4) {
			System.err.println("Son necesarios 4 argumentos con los puertos seriales");
			System.exit(1);
		}
		
		//conexión de la IMU
		System.out.println("Abrimos conexión IMU");
		csi=new ConexionSerialIMU();
		if(!csi.ConectaPuerto(args[1],5)) {
			System.err.println("Problema en conexión serial con la IMU");
			System.exit(1);
		}
		
		//comunicación con GPS
		System.out.println("Abrimos conexión GPS");
//		gpsCon=new SimulaGps(args[0]).getGps();
		try {
			gpsCon=new GPSConnectionTriumph(args[0]);
		} catch (Exception e) {
			System.err.println("Promblema a crear GPSConnection:"+e.getMessage());
			System.exit(1);			
		}
		if(gpsCon==null) {
			System.err.println("No se obtuvo GPSConnection");
			System.exit(1);
		}
		gpsCon.setCsIMU(csi);
		
		
		//Conectamos a RF
		System.out.println("Abrimos conexión LMS");
		try { 		
			manLMS=new ManejaLMS(args[2]);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF");
		}
		
		//Conectamos Carro
		System.out.println("Abrimos conexión al Carro");
		contCarro=new ControlCarro(args[3]);
		
		//Checkbox para navegar
		jcbNavegando=new JCheckBox("Navegando");
		jcbNavegando.setSelected(false);

		//Ventana datos numéricos
		ventNumeros=new JFrame("Datos GPS IMU COCHE");
		JPanel jpCentral=new JPanel();
		ventNumeros.add(jpCentral,BorderLayout.CENTER);
		ventNumeros.getContentPane().add(jcbNavegando,BorderLayout.SOUTH);

		//paneles uno debajo del otro
		jpCentral.setLayout(new BoxLayout(jpCentral,BoxLayout.PAGE_AXIS));
		
		PMGPS=new PanelMuestraGPSData(false);
		PMGPS.actualizaPunto(new GPSData());
		jpCentral.add(PMGPS);

		{
			JPanel jpGPST=new JPanel();
			jpCentral.add(jpGPST);
			
			jlCalidad=new JLabel("Calidad ### %");
			jpGPST.add(jlCalidad);
			gpsCon.addGpsEventListener(PMGPS);
			gpsCon.addGpsEventListener(this);
			
			jlNumPaquetes=new JLabel("Mensajes #######");
			jpGPST.add(jlNumPaquetes);
			
		}


		pmai=new PanelMuestraAngulosIMU();
		pmai.actualizaAngulo(new AngulosIMU(0,0,0,0));
		jpCentral.add(pmai);
		//conecto manejador cuando todas las ventanas están creadas
		csi.addIMUEventListener(pmai);

		pmCoche=new PanelCoche(contCarro);
		jpCentral.add(pmCoche);
		
		ventNumeros.pack();
		ventNumeros.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventNumeros.setVisible(true);

		//elegir fichero
		fc=new JFileChooser(new File("./Rutas"));

		//necestamos leer archivo con la ruta
		do {
			int devuelto=fc.showOpenDialog(ventNumeros);
			if (devuelto!=JFileChooser.APPROVE_OPTION) 
				JOptionPane.showMessageDialog(ventNumeros,
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
		System.out.println("Usando desviación magnética "+Math.toDegrees(desMag));
		
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
		
		//Inicializamos modelos predictivos
		modCoche=new CocheModeloAntiguo();
		cp=new ControlPredictivo(modCoche,Tr,13,3,1.0,(double)periodoMuestreoMili/1000);
		JFrame ventanaPredictivo=new JFrame("Panel Muestra Predictivo");		
		ventanaPredictivo.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pmp=new PanelMuestraPredictivo(cp);
		ventanaPredictivo.add(pmp);
		ventanaPredictivo.setSize(new Dimension(900,700));
		ventanaPredictivo.setVisible(true);

	}

	/** Usamos para actulizar la etiqueta de la calidad */
	public void handleGpsEvent(GpsEvent ev) {
		jlCalidad.setText(String.format("Calidad Enlace: %.0f %%", gpsCon.getCalidadLink()));
		jlNumPaquetes.setText(String.format("Mensajes %10d", gpsCon.getCuentaPaquetesRecibidos()));
	}

	/** Método que ejecuta cada {@link #periodoMuestreoMili} bulce de control del coche mirando los obstáculos con el RF 
	 */
	public void camina() {
		double comandoVelocidad=2;
		modCoche.setVelocidad(comandoVelocidad);
		Thread.currentThread().setPriority(Thread.MAX_PRIORITY);
		long tSig;
		boolean solicitado=false;
		while(true) {
			tSig=System.currentTimeMillis()+periodoMuestreoMili;
			try { 
				if(jcbNavegando.isSelected()) {
//					manLMS.pideBarrido((short)0, (short)180, (short)1);
//					solicitado=true;
					
					//Calculamos el comando
					GPSData pa=gpsCon.getPuntoActualTemporal();
					double[] ptoAct={pa.getXLocal(), pa.getYLocal()};
					double angAct=Math.toRadians(pa.getAngulosIMU().getYaw())+desMag;
					double volante=contCarro.getAnguloVolante();
					modCoche.setPostura(ptoAct[0], ptoAct[1], angAct, volante);
					double comandoVolante = cp.calculaComando(); 
					if (comandoVolante > COTA_ANGULO)
						comandoVolante = COTA_ANGULO;
					if (comandoVolante < -COTA_ANGULO)
						comandoVolante = -COTA_ANGULO;
					//System.out.println("Comando " + comandoVolante);
					contCarro.setAnguloVolante(-comandoVolante);
					//TODO leer velocidad del coche??
					// no hace falta modCoche.setConsignaVolante(comandoVolante);
					modCoche.calculaEvolucion(comandoVolante,comandoVelocidad,periodoMuestreoMili/1000);

					pmp.actualiza();
					
					if(solicitado) {
						BarridoAngular ba=manLMS.recibeBarrido();
						if(pa==null) {
							System.err.println("No hay punto");
							continue;
						}
						//vemos los obstaculos
						double dist=mi.masCercano(ptoAct, angAct, ba);
						pmo.actualiza();
						PMOS.actualiza();
					}
					
					
//					if(Double.isNaN(dist))
//						System.out.println("Estamos fuera del camino");
//					else if(Double.isInfinite(dist))
//						System.out.println("No hay obstáculo");
//					else
//						System.out.println("Distancia="+dist);
					
				}
			} catch (LMSException e) {
				System.err.println("Problemas al obtener barrido en punto "
						+" :"+e.getMessage());
			}
			pmCoche.actualiza();
			pmCoche.repinta();
			//esperamos hasta que hayan pasado miliSeg de ciclo.
			long msSobra=tSig-System.currentTimeMillis();
			System.out.println("Sobra="+msSobra);
			while(System.currentTimeMillis()<tSig)
				try{Thread.sleep(tSig-System.currentTimeMillis());} catch (Exception e) {}	
		}

	}

	/**
	 * @param args Seriales para IMU, GPS, RF y Carro. Si no se pasan de piden interactivamente.
	 */
	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<3) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"GPS","IMU","RF","Coche"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
		NavegaPredictivo na=new NavegaPredictivo(puertos);
		na.camina();
	}

}

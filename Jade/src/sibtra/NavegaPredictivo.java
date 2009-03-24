package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.io.File;

import javax.swing.BoxLayout;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.VentanaCoche;
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
public class NavegaPredictivo implements GpsEventListener, ChangeListener {

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
    double[][] Tr = null;
    private MiraObstaculo mi;
//	private JFrame ventanaPMOS;
    private JFrame ventanaPMO;
    private PanelMiraObstaculo pmo;
//	private PanelMiraObstaculoSubjetivo PMOS;
    private double desMag;
    JCheckBox jcbNavegando;
    JCheckBox jcbFrenando;
    SpinnerNumberModel spFrenado;
    JSpinner JsDistFrenado;
    Coche modCoche;
    ControlPredictivo cp;
    ControlCarro contCarro;
    private PanelMuestraPredictivo pmp;
    private JLabel jlCalidad;
    private VentanaCoche pmCoche;
    private JLabel jlNumPaquetes;
    private JCheckBox jcbUsarRF;
    protected double distRF;
    private int numPaquetesGPS;
	private double gananciaVel = 1;
	private double distFrenado;

    /** Se le han de pasar los 3 puertos series para: IMU, GPS, RF y Coche (en ese orden)*/
    public NavegaPredictivo(String[] args) {
        if (args == null || args.length < 4) {
            System.err.println("Son necesarios 4 argumentos con los puertos seriales");
            System.exit(1);
        }

        //conexión de la IMU
        System.out.println("Abrimos conexión IMU");
        csi = new ConexionSerialIMU();
        if (!csi.ConectaPuerto(args[1], 5)) {
            System.err.println("Problema en conexión serial con la IMU");
            System.exit(1);
        }

        //comunicación con GPS
        System.out.println("Abrimos conexión GPS");
//		gpsCon=new SimulaGps(args[0]).getGps();
        try {
            gpsCon = new GPSConnectionTriumph(args[0]);
        } catch (Exception e) {
            System.err.println("Promblema a crear GPSConnection:" + e.getMessage());
            System.exit(1);
        }
        if (gpsCon == null) {
            System.err.println("No se obtuvo GPSConnection");
            System.exit(1);
        }
        gpsCon.setCsIMU(csi);


        //Conectamos a RF
        System.out.println("Abrimos conexión LMS");
        try {
            manLMS = new ManejaLMS(args[2]);
            manLMS.setDistanciaMaxima(80);
            manLMS.CambiaAModo25();
        } catch (LMSException e) {
            System.err.println("No fue posible conectar o configurar RF");
        }

        //Conectamos Carro
        System.out.println("Abrimos conexión al Carro");
        contCarro = new ControlCarro(args[3]);

        if (contCarro.isOpen() == false) {
            System.err.println("No se obtuvo Conexion al Carro");
            
        }

        //Ventana datos numéricos
        ventNumeros = new JFrame("Datos GPS IMU COCHE");
        JPanel jpCentral = new JPanel();
        ventNumeros.add(jpCentral, BorderLayout.CENTER);
        {   //Parte baja de la ventana
            JPanel jpSur = new JPanel(new FlowLayout(3));
            ventNumeros.getContentPane().add(jpSur, BorderLayout.SOUTH);

            //Checkbox para navegar
            jcbNavegando = new JCheckBox("Navegando");
            jcbNavegando.setSelected(false);
            jpSur.add(jcbNavegando);
            //Checkbox para frenar
            jcbFrenando = new JCheckBox("Frenar");
            jcbFrenando.setSelected(false);
            jpSur.add(jcbFrenando);
            //Spinner para fijar la distancia de frenado
            double value = 5;
            double min = 1;
            double max = 50;
            double step = 0.1;
            spFrenado = new SpinnerNumberModel(value,min,max,step);
            JsDistFrenado = new JSpinner(spFrenado);
            JsDistFrenado.addChangeListener(this);
            jpSur.add(JsDistFrenado);
            //Checkbox para detectar con RF
            jcbUsarRF = new JCheckBox("Usar RF");
            jcbUsarRF.setSelected(false);
            jpSur.add(jcbUsarRF);
        }

        //paneles uno debajo del otro
        jpCentral.setLayout(new BoxLayout(jpCentral, BoxLayout.PAGE_AXIS));

        PMGPS = new PanelMuestraGPSData(false);
        PMGPS.actualizaPunto(new GPSData());
        jpCentral.add(PMGPS);

        {
        	//TODO Usar panel gpsTriumph
            JPanel jpGPST = new JPanel();
            jpCentral.add(jpGPST);

            jlCalidad = new JLabel("Calidad ### %");
            jpGPST.add(jlCalidad);
            gpsCon.addGpsEventListener(PMGPS);
            gpsCon.addGpsEventListener(this);

            jlNumPaquetes = new JLabel("Mensajes #######");
            jpGPST.add(jlNumPaquetes);

        }


        pmai = new PanelMuestraAngulosIMU();
        pmai.actualizaAngulo(new AngulosIMU(0, 0, 0, 0));
        jpCentral.add(pmai);
        //conecto manejador cuando todas las ventanas están creadas
        csi.addIMUEventListener(pmai);

        VentanaCoche vc=new VentanaCoche(contCarro);


        ventNumeros.pack();
        ventNumeros.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventNumeros.setVisible(true);

        //elegir fichero
        fc = new JFileChooser(new File("./Rutas"));

        //necestamos leer archivo con la ruta
        do {
            int devuelto = fc.showOpenDialog(ventNumeros);
            if (devuelto != JFileChooser.APPROVE_OPTION) {
                JOptionPane.showMessageDialog(ventNumeros,
                        "Necesario cargar fichero de ruta",
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            } else {
                gpsCon.loadRuta(fc.getSelectedFile().getAbsolutePath());
            }
        } while (gpsCon.getRutaEspacial() == null);
        //nuestra ruta espacial será la que se cargó
        rutaEspacial = gpsCon.getRutaEspacial();
        desMag = rutaEspacial.getDesviacionM();
        System.out.println("Usando desviación magnética " + Math.toDegrees(desMag));

        //Rellenamos la trayectoria con la nueva versión de toTr,que 
        //introduce puntos en la trayectoria de manera que la separación
        //entre dos puntos nunca sea mayor de la distMax
        double distMax = 0.1;        
        Tr = rutaEspacial.toTr(distMax);


        System.out.println("Longitud de la trayectoria=" + Tr.length);

        mi = new MiraObstaculo(Tr);
        //TODO Activar panel muestra obtácuo subjetivo
//		try {
//			PMOS=new PanelMiraObstaculoSubjetivo(mi,(short)manLMS.getDistanciaMaxima());
//		} catch (LMSException e) {
//			System.err.println("Problema al obtener distancia maxima configurada");
//			System.exit(1);
//		}
//		ventanaPMOS=new JFrame("Mira Obstáculo Subjetivo");
//		ventanaPMOS.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//		ventanaPMOS.getContentPane().add(PMOS,BorderLayout.CENTER);
//		ventanaPMOS.setSize(new Dimension(800,400));
//		ventanaPMOS.setVisible(true);


        ventanaPMO = new JFrame("Mira Obstáculo");
        ventanaPMO.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pmo = new PanelMiraObstaculo(mi);
        ventanaPMO.getContentPane().add(pmo, BorderLayout.CENTER);
        ventanaPMO.setSize(new Dimension(800, 600));
        ventanaPMO.setVisible(true);

        //Inicializamos modelos predictivos
        modCoche = new Coche();
        cp = new ControlPredictivo(modCoche, Tr, 13, 3, 1.0, (double) periodoMuestreoMili / 1000);
        JFrame ventanaPredictivo = new JFrame("Panel Muestra Predictivo");
        ventanaPredictivo.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pmp = new PanelMuestraPredictivo(cp,rutaEspacial);
        ventanaPredictivo.add(pmp);
        ventanaPredictivo.setSize(new Dimension(900, 700));
        ventanaPredictivo.setVisible(true);

    }

    /** Usamos para actulizar la etiqueta de la calidad */
    public void handleGpsEvent(GpsEvent ev) {
        jlCalidad.setText(String.format("Calidad Enlace: %4.0f %%", gpsCon.getCalidadLink()));
        int paquetesAhora = gpsCon.getCuentaPaquetesRecibidos();
        jlNumPaquetes.setText(String.format("Mensajes %10d", paquetesAhora));
    }
    
    /**
     * Método para decidir la consigna de velocidad para cada instante. Se programan tres estados:
     * Arrancando,navegando y frenando.
     * En el estado navegando se tiene en cuenta el error en la orientación 
     * @return
     */
    public double calculaConsignaVel(){
        double consigna = 0;
        double velocidadMax = 3;
        double refVelocidad;
        double errorOrientacion;       
        int indMin = ControlPredictivo.calculaDistMin(Tr,modCoche.getX(),modCoche.getY());       
        errorOrientacion = cp.getOrientacionDeseada() - modCoche.getTita();
        if (Tr[indMin][3]>velocidadMax){
            refVelocidad = velocidadMax;
        }else
            refVelocidad = Tr[indMin][3]; 
        consigna = refVelocidad - errorOrientacion*gananciaVel;
        if (consigna <= 0)
            consigna = 0.5;       
        return consigna; 
     }
    /**
     * Recorre la trayectoria desde el punto más cercano al coche y mide la distancia de 
     * frenado. Devuelve el índice del punto que se encuentra a esa distancia
     * @param distFrenado Distancia en metros a la que se desea que el coche se detenga
     * @return Índice del punto que se encuentra a la distancia de frenado
     */
    public int buscaPuntoFrenado(double distFrenado){
    	int indCercano = ControlPredictivo.calculaDistMin(Tr, modCoche.getX(),modCoche.getY());
    	double dist = 0;
    	int i = 0;    	
    	for (i=indCercano;dist<distFrenado;i++){
    		double dx=Tr[i][0]-Tr[(i+1)%Tr.length][0];
            double dy=Tr[i][1]-Tr[(i+1)%Tr.length][1];
            dist = dist + Math.sqrt(dx*dx+dy*dy);
    	}
    	return i;
    }
    /**
     * Calcula la distancia a la que se encuentra el punto en el que se quiere que el coche
     * se detenga
     * @param puntoFrenado Índice del punto de la trayectoria donde se desea que el coche se pare
     * @return Distancia en metros a la que se encuentra el punto en el que se desea que el
     * coche se detenga
     */
    public double mideDistanciaFrenado(int puntoFrenado){
    	double distFrenado=0;
    	int indCercano = ControlPredictivo.calculaDistMin(Tr, modCoche.getX(),modCoche.getY());    	   	
    	for (int i=indCercano;i<puntoFrenado;i++){
    		double dx=Tr[i][0]-Tr[(i+1)%Tr.length][0];
            double dy=Tr[i][1]-Tr[(i+1)%Tr.length][1];
            distFrenado = distFrenado + Math.sqrt(dx*dx+dy*dy);
    	}
    	return distFrenado;
    }
    /**
     * calcula la rampa decreciente de consignas de velocidad para realizar el frenado
     * del coche
     * @param velocidadActual Velocidad instantanea en metros por segundo del coche 
     * @param distFrenado Distancia en metros a la que se desea que el coche se detenga
     * @param numPuntos cantidad de puntos del perfil. Coincidirá con el horizonte de predicción
     * en el caso de que el perfil de velocidad sea la entrada para el controlador predictivo
     * @param T Periodo de muestreo del sistema
     * @return Perfil de velocidad de frenado
     */
    public double[] calculaPerfilVelocidad(double velocidadActual,double distFrenado,int numPuntos,double T){
    	double pendiente = -velocidadActual/distFrenado;
    	double c = -pendiente*distFrenado;
    	double t = T;
    	double[] perfilVelocidad= new double[numPuntos];
    	for(int i=0;i<numPuntos+1;i++){    		
    		perfilVelocidad[i] = pendiente*t + c;
    		t = t + T;
    	}    	
    	return perfilVelocidad;
    }
    
    /** Método que ejecuta cada {@link #periodoMuestreoMili} bulce de control del coche mirando los obstáculos con el RF 
     */
    public void camina() {
        Thread thRF = new Thread() {

            private long periodoMuestreoMiliRF = 500;

            public void run() {
                long tSig;
                boolean solicitado = false;
                while (true) {
                    tSig = System.currentTimeMillis() + periodoMuestreoMiliRF;
                    try {
                        if (jcbUsarRF.isSelected()) {
                            manLMS.pideBarrido((short) 0, (short) 180, (short) 1);
                            BarridoAngular ba = manLMS.recibeBarrido();

                            //Calculamos el comando
                            GPSData pa = gpsCon.getPuntoActualTemporal();
                            double[] ptoAct = {pa.getXLocal(), pa.getYLocal()};
                            double angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + desMag;
                            distRF = mi.masCercano(ptoAct, angAct, ba);
                            pmo.actualiza();
//							PMOS.actualiza();


//							if(Double.isNaN(dist))
//								System.out.println("Estamos fuera del camino");
//							else if(Double.isInfinite(dist))
//								System.out.println("No hay obstáculo");
//							else
//								System.out.println("Distancia="+dist);

                        }
                    } catch (LMSException e) {
                        System.err.println("Problemas al obtener barrido en punto " + " :" + e.getMessage());
                    }
                    //TODO poner RF a todo lo que da
                    long msSobra = tSig - System.currentTimeMillis();
                    if (msSobra < 0) {
                        System.out.println("Sobra RF =" + msSobra);
                    }
                    while (System.currentTimeMillis() < tSig) {
                        try {
                            Thread.sleep(tSig - System.currentTimeMillis());
                        } catch (Exception e) {
                        }
                    }
                }

            }
        };

        thRF.start();
        double comandoVelocidad;        
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY);
        long tSig;
        while (true) {
            tSig = System.currentTimeMillis() + periodoMuestreoMili;
            if (jcbNavegando.isSelected()) {

                //Calculamos el comando
                GPSData pa = gpsCon.getPuntoActualTemporal();
                double[] ptoAct = {pa.getXLocal(), pa.getYLocal()};
                double angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + desMag;
                double volante = contCarro.getAnguloVolante();
                // Con esta linea realimentamos la información de los sensores al modelo
                // se puede incluir tambien la posicion del volante añadiendo el parámetro
                // volante a la invocación de setPostura
                modCoche.setPostura(ptoAct[0], ptoAct[1], angAct);
                double comandoVolante = cp.calculaComando();
                double consignaVelocidad = calculaConsignaVel();
                System.out.println(consignaVelocidad);
                if (comandoVolante > COTA_ANGULO) {
                    comandoVolante = COTA_ANGULO;
                }
                if (comandoVolante < -COTA_ANGULO) {
                    comandoVolante = -COTA_ANGULO;
                //System.out.println("Comando " + comandoVolante);
                }
                contCarro.setAnguloVolante(-comandoVolante);
                contCarro.setConsignaAvanceMS(consignaVelocidad);
//				contCarro.Avanza(120);
                //TODO leer la posición del volante
                comandoVelocidad = contCarro.getVelocidadMS();
                // no hace falta modCoche.setConsignaVolante(comandoVolante);
                modCoche.calculaEvolucion(comandoVolante, comandoVelocidad, periodoMuestreoMili / 1000);

                pmp.actualiza();
            }
            //para mostrar si se dejaron de recibir paquetes del GPS
            int paquetesAhora = gpsCon.getCuentaPaquetesRecibidos();
            if (numPaquetesGPS == paquetesAhora) {
                PMGPS.actualizaPunto(null);  //desabilitará todas las labels
            }
            numPaquetesGPS = paquetesAhora;

            /* no hace falta porque hay un thread que refresca automaticamente
             * pmCoche.actualiza(); 
            pmCoche.repinta(); */
            //esperamos hasta que hayan pasado miliSeg de ciclo.
            long msSobra = tSig - System.currentTimeMillis();
            if (msSobra < 0) {
                System.out.println("Sobra=" + msSobra);
            }
            while (System.currentTimeMillis() < tSig) {
                try {
                    Thread.sleep(tSig - System.currentTimeMillis());
                } catch (Exception e) {
                }
            }
        }

    }

    /**
     * @param args Seriales para IMU, GPS, RF y Carro. Si no se pasan de piden interactivamente.
     */
    public static void main(String[] args) {
        String[] puertos;
        if (args == null || args.length < 3) {
            //no se han pasado argumentos, pedimos los puertos interactivamente
            String[] titulos = {"GPS", "IMU", "RF", "Coche"};
            puertos = new EligeSerial(titulos).getPuertos();
            if (puertos == null) {
                System.err.println("No se asignaron los puertos seriales");
                System.exit(1);
            }
        } else {
            puertos = args;
        }
        NavegaPredictivo na = new NavegaPredictivo(puertos);
        na.camina();
    }

	public void stateChanged(ChangeEvent e) {
		// TODO Auto-generated method stub
		if(e.getSource()==spFrenado){
			distFrenado = spFrenado.getNumber().doubleValue();
		}
		
	}
}

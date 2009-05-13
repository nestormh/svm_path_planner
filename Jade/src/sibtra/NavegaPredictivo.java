																																																																																																																																																																																																																																																																																																																					package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.PanelCarro;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.GPSData;
import sibtra.gps.PanelGPSTriumph;
import sibtra.gps.Ruta;
import sibtra.imu.AngulosIMU;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelMuestraAngulosIMU;
import sibtra.lms.BarridoAngular;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.predictivo.Coche;
import sibtra.predictivo.ControlPredictivo;
import sibtra.predictivo.PanelMuestraPredictivo;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.util.EligeSerial;
import sibtra.util.UtilCalculos;

/**
 * Para realizar la navegación controlando el coche con @link {@link ControlPredictivo}
 *  detectando obstáculos con el RF.
 * @author alberto
 *
 */
public class NavegaPredictivo implements ActionListener {

    /** Milisegundos del ciclo */
    private static final long periodoMuestreoMili = 200;
    private static final double COTA_ANGULO = Math.toRadians(30);
    /** Si la velocidad es más baja que este umbral se deja de mandar comandos al volante*/
	private static final double umbralMinimaVelocidad = 0.2;
	/** Distancia a la que el coche empieza a frenar si se encuentra un obstáculo a 
	 * menos de esa distancia*/
	private static final double distanciaSeguridad = 10;
	/** Distancia a la que idealmente se detendrá el coche del obstáculo*/
	private static final double margenColision = 2;
    private ConexionSerialIMU csi;
    private GPSConnectionTriumph gpsCon;
    private ManejaLMS manLMS;
    private JFrame ventNumeros;
    private PanelGPSTriumph pgt;
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
    JSpinner jsDistFrenado;
    Coche modCoche;
    ControlPredictivo cp;
    ControlCarro contCarro;
    private PanelMuestraPredictivo pmp;
    private PanelCarro pmCoche;
    private JCheckBox jcbUsarRF;
    protected double distRF = 80;
    /** Regula la velocidad que se resta a la consigna principal de velocidad por 
     * errores en la orientación*/
	private double gananciaVel = 2;
	/** Cuando se manda a frenar la función {@link buscaPuntoFrenado} devuelve en esta variable 
	 * el punto que se encuentra a la distancia de frenado*/
	private int puntoFrenado=-1;
	private GPSData centroToTr;
	private SpinnerNumberModel spGananciaVel;
	private JSpinner jsGananciaVel;
	/** Regula la velocidad que se resta a la consigna principal de velocidad por 
     * errores en la posición lateral*/
	private double gananciaLateral=1;
	/** Pendiente de la rampa de frenado para la parada total */
	private double pendienteFrenado=1.0;

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
        try {
            gpsCon = new GPSConnectionTriumph(args[0]);
        } catch (Exception e) {
            System.err.println("Problema a crear GPSConnection:" + e.getMessage());
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
            manLMS.setResolucionAngular((short)100);
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
        {
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
        		jcbFrenando.addActionListener(this);
        		jpSur.add(jcbFrenando);
        		//Spinner para fijar la distancia de frenado
        		double value = 5;
        		double min = 1;
        		double max = 50;
        		double step = 0.1;
        		spFrenado = new SpinnerNumberModel(value,min,max,step);
        		jsDistFrenado = new JSpinner(spFrenado);
        		jpSur.add(jsDistFrenado);
        		// Spinner para fijar la ganancia del cálculo d la consigna de Velocidad
        		jpSur.add(new JLabel("Ganancia Velocidad"));
        		spGananciaVel = new SpinnerNumberModel(2,0.1,20,0.1);
        		jsGananciaVel = new JSpinner(spGananciaVel);
        		jpSur.add(jsGananciaVel);

        		//Checkbox para detectar con RF
        		jcbUsarRF = new JCheckBox("Usar RF");
        		jcbUsarRF.setSelected(false);
        		jpSur.add(jcbUsarRF);
        	}

        	//paneles uno debajo del otro
        	jpCentral.setLayout(new BoxLayout(jpCentral, BoxLayout.PAGE_AXIS));

        	//Panel del GPS
        	pgt = new PanelGPSTriumph(gpsCon);
        	pgt.setBorder(BorderFactory.createTitledBorder("GPS"));
        	pgt.actualizaGPS(new GPSData());
        	jpCentral.add(pgt);

        	//Panel de la Imu
        	pmai = new PanelMuestraAngulosIMU();
        	pmai.setBorder(BorderFactory.createTitledBorder("IMU"));
        	pmai.actualizaAngulo(new AngulosIMU(0, 0, 0, 0));
        	jpCentral.add(pmai);
        	
        	//Panel del Coche
        	pmCoche=new PanelCarro(contCarro);
        	pmCoche.setBorder(BorderFactory.createTitledBorder("COCHE"));
        	jpCentral.add(pmCoche);

        	ventNumeros.pack();
        	ventNumeros.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        	ventNumeros.setVisible(true);
        	
        	//Tread para refrescar los paneles de la ventana
            Thread thRefresco = new Thread() {
            	/** Milisegundos del periodo de actualización */
            	private long milisPeriodo=500;

                public void run() {
            		while (true){
//            			pgt.setEnabled(true);
            			//GPS
            			pgt.actualizaGPS(gpsCon.getPuntoActualTemporal());
            			pgt.repinta();
            			//IMU
        				pmai.actualizaAngulo(csi.getAngulo());
        				pmai.repinta();
        				//Coche
        				pmCoche.actualizaCarro();
        				pmCoche.repinta();

        				try{Thread.sleep(milisPeriodo);} catch (Exception e) {}	
            		}
                }
            };
            thRefresco.start();
        }


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
        // MOstrar coodenadas del centro del sistema local
        centroToTr = rutaEspacial.getCentro();
        System.out.println("centro de la Ruta Espacial " + centroToTr);
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
        ventanaPMO.pack();
        ventanaPMO.setSize(new Dimension(800, 600));
        ventanaPMO.setVisible(true);

        //Inicializamos modelos predictivos
        modCoche = new Coche();
        cp = new ControlPredictivo(modCoche, Tr, 13, 4, 2.0, (double) periodoMuestreoMili / 1000);
        JFrame ventanaPredictivo = new JFrame("Panel Muestra Predictivo");
        ventanaPredictivo.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pmp = new PanelMuestraPredictivo(cp,rutaEspacial);
        ventanaPredictivo.add(pmp);
        ventanaPredictivo.setSize(new Dimension(900, 700));
        ventanaPredictivo.setVisible(true);

    }
    
    /**
     * Método para decidir la consigna de velocidad para cada instante.
     * Se tiene en cuenta el error en la orientación y el error lateral para reducir la 
     * consigna de velocidad. 
     * @return
     */
    public double calculaConsignaVel(double consignaAnt){
        double consigna = 0;
        double velocidadMax = 2.5;
        double VelocidadMinima = 1;
        double refVelocidad;
        double errorOrientacion;      
        double errorLateral;
        int indMin = UtilCalculos.indiceMasCercano(Tr,modCoche.getX(),modCoche.getY());
        double dx = Tr[indMin][0]-modCoche.getX();
        double dy = Tr[indMin][1]-modCoche.getY();
        errorLateral = Math.sqrt(dx*dx + dy*dy);
//        errorOrientacion = cp.getOrientacionDeseada() - modCoche.getTita();
        errorOrientacion = Tr[indMin][2] - modCoche.getTita();
//        System.out.println("Error en la orientación "+errorOrientacion);
        if (Tr[indMin][3]>velocidadMax){
            refVelocidad = velocidadMax;
        }else
            refVelocidad = Tr[indMin][3]; 
        consigna = refVelocidad - Math.abs(errorOrientacion)*gananciaVel - Math.abs(errorLateral)*gananciaLateral;        
        if (consigna-consignaAnt >=0.1){
        	consigna = consignaAnt + 0.1;
        	System.out.println("Demasiado incremento en la consigna");
        }
/*      Solo con esta condición el coche no se detiene nunca,aunque la referencia de la 
 * 		ruta sea cero*/
//        if (consigna <= 1){
//            consigna = 1;
        if (consigna <= VelocidadMinima && refVelocidad >= VelocidadMinima){
        /*Con esta condición se contempla el caso de que la consigna sea < 0*/
            consigna = VelocidadMinima;
        }else if (consigna <= VelocidadMinima && refVelocidad <= VelocidadMinima)
        /* De esta manera si la velocidad de la ruta disminuye hasta cero el coche se 
        detiene, en vez de seguir a velocidad mínima como ocurría antes. En este caso también
        está contemplado el caso de que la consigna sea < 0*/
        	consigna = refVelocidad;
        return consigna; 
     }
    /**
     * Recorre la trayectoria desde el punto más cercano al coche y mide la distancia de 
     * frenado. Devuelve el índice del punto que se encuentra a esa distancia
     * @param distFrenado Distancia en metros a la que se desea que el coche se detenga
     * @return Índice del punto que se encuentra a la distancia de frenado
     */
    public int buscaPuntoFrenado(double distFrenado){
    	int indCercano = UtilCalculos.indiceMasCercano(Tr, modCoche.getX(),modCoche.getY());
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
    	int indCercano = UtilCalculos.indiceMasCercano(Tr, modCoche.getX(),modCoche.getY());    	   	
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
    public double calculaPerfilVelocidad(double velocidadActual,double distFrenado,double T){
    	double pendiente = -velocidadActual/distFrenado;
    	double c = -pendiente*distFrenado;
    	double consignaVelocidad = pendiente*T + c;       	
    	return consignaVelocidad;
    }
    
	/** Método que ejecuta cada {@link #periodoMuestreoMili} bucle de control del coche mirando los obstáculos con el RF 
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
        double velocidadActual;        
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY);
        long tSig;
        double consignaVelAnt = 0;
        while (true) {
            tSig = System.currentTimeMillis() + periodoMuestreoMili;
            if (jcbNavegando.isSelected()) {

                //Calculamos el comando            	
                GPSData pa = gpsCon.getPuntoActualTemporal();
                GPSData centroactual = gpsCon.getBufferEspacial().getCentro();
                if (centroactual.getAltura()!=centroToTr.getAltura()
                		|| centroactual.getLatitud()!=centroToTr.getLatitud()
                		|| centroactual.getLongitud()!=centroToTr.getLongitud())
                	System.err.println("El centro es diferente!!! " + centroToTr 
                			+"!= "+  gpsCon.getBufferEspacial().getCentro());
                double[] ptoAct = {pa.getXLocal(), pa.getYLocal()};
                double angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + desMag;
                double volante = contCarro.getAnguloVolante();
                // Con esta linea realimentamos la información de los sensores al modelo
                // se puede incluir tambien la posicion del volante añadiendo el parámetro
                // volante a la invocación de setPostura
                //TODO Realimentar posición del volante
                modCoche.setPostura(ptoAct[0], ptoAct[1], angAct);
                double comandoVolante = cp.calculaComando();                
                if (comandoVolante > COTA_ANGULO) {
                    comandoVolante = COTA_ANGULO;
                }
                if (comandoVolante < -COTA_ANGULO) {
                    comandoVolante = -COTA_ANGULO;
                //System.out.println("Comando " + comandoVolante);
                }
                double consignaVelocidad;
                velocidadActual = contCarro.getVelocidadMS();
                //Cuando está casi parado no tocamos el volante
                if (velocidadActual >= umbralMinimaVelocidad)
                	contCarro.setAnguloVolante(-comandoVolante);
                
            	consignaVelocidad = calculaConsignaVel(consignaVelAnt); 
            	//Si se pulsa la checkbox de frenar
                if (puntoFrenado!=-1){
            		double distFrenado = mideDistanciaFrenado(puntoFrenado);
            		double velRampa=distFrenado*pendienteFrenado;
            		//se contempla el caso de que se esté frenando porque se ha pulsado
            		//la checkbox Frenar y a la vez el RF detecte un obstáculo
            		if (jcbUsarRF.isSelected() && (distRF <= distanciaSeguridad)){
            			/* Resto margenColision a distRf para que el coche se detenga a 
            			 * esa distancia del obstáculo */
            			double velRampaRF = (distRF-margenColision)*pendienteFrenado; 
            			velRampa = Math.min(velRampa,velRampaRF);
            		}
            		// Nos quedamos con la velocidad menor, la más restrictiva
            		consignaVelocidad=Math.min(consignaVelocidad, velRampa);
            		System.out.println("Punto frenado a "+distFrenado+" vel. rampa "+ velRampa);
                } else if (jcbUsarRF.isSelected() && (distRF <= distanciaSeguridad)){
                    // Si el RF detecta un obstáculo a menos de la dist de seguridad
                	double velRampa = (distRF-margenColision)*pendienteFrenado;
                	consignaVelocidad=Math.min(consignaVelocidad, velRampa);
                }
                System.out.println(consignaVelocidad);
                consignaVelAnt = consignaVelocidad;
                contCarro.setConsignaAvanceMS(consignaVelocidad);			
                modCoche.calculaEvolucion(comandoVolante, velocidadActual, periodoMuestreoMili / 1000);
                pmp.actualiza();
            }

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
    public void actionPerformed(ActionEvent e) {
		if (e.getSource() == jcbFrenando){
			if(jcbFrenando.isSelected()){
				// Se acaba de seleccionar
				double distFrenado = spFrenado.getNumber().doubleValue();
				puntoFrenado = buscaPuntoFrenado(distFrenado);
				jsDistFrenado.setEnabled(false);
			}else{
				// Se acaba de desactivar
				jsDistFrenado.setEnabled(true);
				puntoFrenado = -1;
			}
			if(e.getSource() == spGananciaVel){
				gananciaVel = spGananciaVel.getNumber().doubleValue();
			}		
		}
		if (e.getSource() == jcbNavegando){
			if (jcbNavegando.isSelected())
				cp.iniciaNavega();
		}
		if (e.getSource() == jcbUsarRF){
			if(!jcbUsarRF.isSelected()){
				//Cuando se desactiva la checkbox del rangeFinder la distancia se
				//se pone al máximo.
				distRF = 80;
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


}

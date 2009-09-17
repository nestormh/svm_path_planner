package sibtra.predictivo;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Label;
import java.awt.event.MouseEvent;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.log.Logger;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerDouble;
import sibtra.log.LoggerFactory;
import sibtra.log.VentanaLoggers;
import sibtra.util.PanelFlow;
import sibtra.util.PanelMuestraTrayectoria;

/**
 * Panel para mostrar la información del control predictivo y poder depurarlo.
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelMuestraPredictivo extends PanelMuestraTrayectoria implements ChangeListener {

    /** Largo de la flecha que marcará orientación final de la predicción */
    private static final double largoFlecha = 2;
    /** Valor máximo de la barra de progreso */
    private static final int pbMax = 100;
    private static final double COTA_ANGULO = Math.toRadians(30);
    /** Puntero al controlador predicctivo a mostrar */
    ControlPredictivo CP = null;

    /** Ruta para poder cambiar la distancia maxima entre puntos */
    Ruta rutaAux=null;
    
    /** Barra progreso para comando a la izquierda */
    private JProgressBar jpbComandoI;
    /** Label que presentará el comando calculado */
    private JLabel jlComando;
    /** Barra progreso para comando a la izquierda */
    private JProgressBar jpbComandoD;
    private JLabel jlDistancia;
    private SpinnerNumberModel jsModHPred;
    private SpinnerNumberModel jsModHCont;
    private SpinnerNumberModel jsModLanda;
    private SpinnerNumberModel jsModDistMax;
    private SpinnerNumberModel jsModAlpha;
    private SpinnerNumberModel jsModPesoError;
	private JSpinner jsDistMax;
	private JSpinner jsHorPred;
	private JSpinner jsHorCont;
	private JSpinner jsLanda;
	private JSpinner jsAlpha;
	private JSpinner jsPesoError;

    /** Constructor necesita el controlador predictivo */
    public PanelMuestraPredictivo(ControlPredictivo contPredic, Ruta rutaIn) {
    	this(contPredic);
    	rutaAux=rutaIn;
    }
    
    public PanelMuestraPredictivo(ControlPredictivo contPredic) {
        super();
        {//nuevo panel para añadir debajo
//            JPanel jpPre = new JPanel(new FlowLayout(FlowLayout.LEADING));
            JPanel jpPre = new PanelFlow();
            //barra de progreso para el comando
            jpbComandoI = new JProgressBar(-pbMax, 0);
            jpPre.add(jpbComandoI);

            jlComando = new JLabel("+##.##º");
            jlComando.setHorizontalAlignment(SwingConstants.TRAILING);
            //jlComando.setMinimumSize(new Dimension(50, 20));
            jlComando.setPreferredSize(new Dimension(60, 20));
            jpPre.add(jlComando);

            //barra de progreso para el comando derecho
            jpbComandoD = new JProgressBar(0, pbMax);
            jpPre.add(jpbComandoD);

            jlDistancia = new JLabel("DL: +###.##");
            //jlDistancia.setMinimumSize(new Dimension(60, 20));
            jlDistancia.setPreferredSize(new Dimension(80, 20));
            //jlDistancia.setSize(new Dimension(60, 20));
            jlDistancia.setHorizontalAlignment(SwingConstants.TRAILING);
            jpPre.add(jlDistancia);

            jpPre.add(new Label("H Pred"));
            jsModHPred = new SpinnerNumberModel(1, 1, 25, 1);
            jsHorPred = new JSpinner(jsModHPred);
            jsModHPred.addChangeListener(this);
            jpPre.add(jsHorPred);

            jpPre.add(new Label("H Cont"));
            jsModHCont = new SpinnerNumberModel(1, 1, 25, 1);
            jsHorCont = new JSpinner(jsModHCont);
            jsModHCont.addChangeListener(this);
            jpPre.add(jsHorCont);

            jpPre.add(new Label("Landa"));
            jsModLanda = new SpinnerNumberModel(0, 0, 100, 0.1);
            jsLanda = new JSpinner(jsModLanda);
            jsModLanda.addChangeListener(this);
            jpPre.add(jsLanda);

            jpPre.add(new Label("Dist Max"));
            jsModDistMax = new SpinnerNumberModel(0.7, 0.05, 1, 0.05);
            jsDistMax = new JSpinner(jsModDistMax);
            jsModDistMax.addChangeListener(this);
            jsDistMax.setEnabled(false); //solo se habilita si hay ruta axiliar
            jpPre.add(jsDistMax);

            jpPre.add(new Label("Alpha"));          
            jsModAlpha = new SpinnerNumberModel(1.05,0.01,2,0.01);
            jsAlpha = new JSpinner(jsModAlpha);
            jsModAlpha.addChangeListener(this);
            jpPre.add(jsAlpha);
            
            jpPre.add(new Label("Peso Error"));          
            jsModPesoError = new SpinnerNumberModel(0.01,0.01,10,0.01);
            jsPesoError = new JSpinner(jsModPesoError);
            jsModPesoError.addChangeListener(this);
            jpPre.add(jsPesoError);
            
//            jpPre.setMinimumSize(new Dimension(Short.MAX_VALUE, 40));
//			jpPre.setBorder(BorderFactory.createCompoundBorder(
//	                   BorderFactory.createLineBorder(Color.blue),
//	                   jpPre.getBorder()));

            add(jpPre);
        }
        setControlP(contPredic);
    }

    /** Fija los nuevos valores para {@link #CP} y {@link #rutaAux} y actualiza la presentación.
     * Uno o los dos pueden ser null.
     */
    public void setControlPyRuta(ControlPredictivo CP,Ruta ruta) {
    	this.CP=CP;
    	this.rutaAux=ruta;
    	jsDistMax.setEnabled(this.rutaAux!=null);
    	if(CP!=null) setTrayectoria(CP.ruta); 
    	actualiza();
    }
    
    /** Fija los nuevos valores para {@link #CP} y {@link #rutaAux} y actualiza la presentación.
     * Uno o los dos pueden ser null.
     */
    public void setControlP(ControlPredictivo CP) {
    	this.CP=CP;
    	if(CP!=null) setTrayectoria(CP.ruta); 
    	jsDistMax.setEnabled(this.rutaAux!=null);
    	actualiza();
    }
    
    /** Lo que añadimos al panel */
    protected void cosasAPintar(Graphics g0) {
    	super.cosasAPintar(g0);
    	if(CP==null) return; //si no hay control predictivo, ¡no pintamos nada! :-)
        //colocamos el coche en su posición actual
        situaCoche(CP.carroOriginal.getX(), CP.carroOriginal.getY(), CP.carroOriginal.getTita());
        super.cosasAPintar(g0);
        Graphics2D g = (Graphics2D) g0;

        //pintamos la trayectoria predicha
        GeneralPath gpPred = pathArrayXY(CP.prediccionPosicion);
        if (gpPred != null) {
            g.setStroke(new BasicStroke());
            g.setColor(Color.WHITE);
            g.draw(gpPred);
        }

        //pintamos la orientación final
        {
            double[] pini = CP.prediccionPosicion[CP.horPrediccion - 1];
            double ori = CP.predicOrientacion[CP.horPrediccion - 1];
            g.setStroke(new BasicStroke());
            g.setColor(Color.RED);
            double[] pfin = {pini[0] + largoFlecha * Math.cos(ori), pini[1] + largoFlecha * Math.sin(ori)};
            g.draw(new Line2D.Double(point2Pixel(pini), point2Pixel(pfin)));

        }
    }

    /** Atiende cambios en los spiners de los parámetros del controlador */
    public void stateChanged(ChangeEvent ce) {
        if (ce.getSource() == jsModHPred) {
            CP.setHorPrediccion(jsModHPred.getNumber().intValue());
        }
        if (ce.getSource() == jsModHCont) {
            int hc = jsModHCont.getNumber().intValue();
            int hp = jsModHPred.getNumber().intValue();
            if (hc > hp) {
                jsModHCont.setValue(hp);
            }
            CP.setHorControl(hc);
        }
        if (ce.getSource() == jsModLanda) {
            CP.setLanda(jsModLanda.getNumber().doubleValue());
        }
        if (ce.getSource() == jsModDistMax) {
            double distMax = jsModDistMax.getNumber().doubleValue();
            CP.setRuta(new Trayectoria(rutaAux,distMax));
        }
        if (ce.getSource() == jsModAlpha){
        	double alpha = jsModAlpha.getNumber().doubleValue();
        	CP.setAlpha(alpha);
        }
        if (ce.getSource() == jsModPesoError){
        	double pesoError = jsModPesoError.getNumber().doubleValue();
        	CP.setPesoError(pesoError);
        }
    }

    
    
    /** programa el repintado del panel actulizando los valores de los spiners, etiquetas etc.*/
    public void actualiza() {
    	boolean hayCP=(CP!=null);
    	if(hayCP) {
    		//barra de progreso con el comando
    		if (CP.comandoCalculado > 0) {
    			jpbComandoI.setValue(pbMax);
    			jpbComandoD.setValue((int) (CP.comandoCalculado * pbMax / (Math.PI / 4)));
    		} else {
    			jpbComandoD.setValue(0);
    			jpbComandoI.setValue((int) (CP.comandoCalculado * pbMax / (Math.PI / 4)));
    		}
    		//texto con el comando
    		jlComando.setText(String.format("%+04.2fº", Math.toDegrees(CP.comandoCalculado)));
    		//texto con la distancia
    		jlDistancia.setText(String.format("DL=%+04.2f", CP.distanciaLateral));

    		//reflejamos valores usados por controlador
    		jsModHCont.setValue(CP.horControl);
    		jsModHPred.setValue(CP.horPrediccion);
    		jsModLanda.setValue(CP.landa);
    		jsModPesoError.setValue(CP.getPesoError());
    		setTrayectoria(CP.ruta);
    	} 
    	jpbComandoD.setEnabled(hayCP);
    	jpbComandoI.setEnabled(hayCP);
    	jlComando.setEnabled(hayCP);
    	jlDistancia.setEnabled(hayCP);
    	jsAlpha.setEnabled(hayCP);
    	jsHorCont.setEnabled(hayCP);
    	jsHorPred.setEnabled(hayCP);
    	jsLanda.setEnabled(hayCP);
    	jsPesoError.setEnabled(hayCP);
    	jsDistMax.setEnabled(hayCP && (rutaAux!=null));
        super.actualiza();
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        final Coche carroOri = new Coche();
        final CocheModeloAntiguo carroViejo = new CocheModeloAntiguo();
        final VentanaLoggers vl; //ventana de los loggers
        double vel = 2;
        double consVolante = 0;
        carroOri.setVelocidad(vel);
        carroOri.setConsignaVolante(consVolante);
        carroViejo.setVelocidad(vel);
        carroViejo.setConsignaVolante(consVolante);
        int horPredic = 10;
        int horCont = 3;
        double paramLanda = 1;
        double paramTs = 0.2;

        Ruta re;
        Trayectoria rutaPruebaRellena;
//        String fichero = "Rutas/Parq20";
        String fichero = "Rutas/Iter1";
//          String fichero = "Rutas/casa23";
//        String fichero = "Rutas/Parq0121_1";
        try {
            File file = new File(fichero);
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
            re = (Ruta) ois.readObject();
            ois.close();
            double distMax = 0.5;
            rutaPruebaRellena = new Trayectoria(re,distMax);
            System.out.println(rutaPruebaRellena.length());
            System.out.println("Abrimos el fichero");

        } catch (IOException ioe) {
            re = new Ruta();
            rutaPruebaRellena = null;
            System.err.println("Error al abrir el fichero " + fichero);
            System.err.println(ioe.getMessage());
        } catch (ClassNotFoundException cnfe) {
            re = new Ruta();
            rutaPruebaRellena = null;
            System.err.println("Objeto leído inválido: " + cnfe.getMessage());
        }


        carroOri.setPostura(-1, -1, 0.5, 0.0);
        carroViejo.setPostura(-1, -1, 0.5, 0.0);

        ControlPredictivo controlador = new ControlPredictivo(carroOri, rutaPruebaRellena,
                horPredic, horCont, paramLanda, paramTs);
        //ventana
        JFrame ventana = new JFrame("Panel Muestra Predictivo");
        ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        PanelMuestraPredictivo pmp = new PanelMuestraPredictivo(controlador, re) {

            /** Evento cuando se pulsó el ratón con el SHIFT, establece la posición deseada */
            MouseEvent evenPos;

            /**
             * Sólo nos interesan pulsaciones del boton 1. 
             * Con CONTROL para determinar posición y orientación. Sin nada para hacer zoom.
             * @see #mouseReleased(MouseEvent)
             */
            public void mousePressed(MouseEvent even) {
                evenPos = null;
                if (even.getButton() == MouseEvent.BUTTON1 && (even.getModifiersEx() & MouseEvent.CTRL_DOWN_MASK) != 0) {
                    //Punto del coche
                    Point2D.Double nuevaPos = pixel2Point(even.getX(), even.getY());
                    System.out.println("Pulsado Boton 1 con CONTROL " + even.getButton() + " en posición: (" + even.getX() + "," + even.getY() + ")" + "  (" + nuevaPos.getX() + "," + nuevaPos.getY() + ")  ");
                    evenPos = even;
                    return;
                }
                //al del padre lo llamamos al final
                super.mousePressed(even);
            }

            /**
             * Las pulsaciones del boton 1 con CONTROL para determinar posición y orientación.
             * Termina el trabajo empezado en {@link #mousePressed(MouseEvent)}
             */
            public void mouseReleased(MouseEvent even) {
                if (even.getButton() == MouseEvent.BUTTON1 && (even.getModifiersEx() & MouseEvent.CTRL_DOWN_MASK) != 0 && evenPos != null) {
                    System.out.println("Soltado con Control Boton " + even.getButton() + " en posición: (" + even.getX() + "," + even.getY() + ")");
                    //Creamos rectángulo si está suficientemente lejos
                    if (Math.abs(even.getX() - evenPos.getX()) > 50 || Math.abs(even.getY() - evenPos.getY()) > 50) {
                        Point2D.Double nuevaPos = pixel2Point(evenPos.getX(), evenPos.getY());
                        Point2D.Double posAngulo = pixel2Point(even.getX(), even.getY());
                        double yaw = Math.atan2(nuevaPos.getY() - posAngulo.getY(), nuevaPos.getX() - posAngulo.getX());
                        carroOri.setPostura(nuevaPos.getX(), nuevaPos.getY(), yaw, 0.0);
                        actualiza();

                    }
                    return;
                }
                //al final llamamos al del padre
                super.mouseReleased(even);
            }
        };

        ventana.add(pmp);
        JCheckBox jcbCaminar = new JCheckBox("Caminar");
        jcbCaminar.setSelected(false);
        ventana.add(jcbCaminar, BorderLayout.PAGE_END);
        ventana.setSize(new Dimension(900, 700));
        ventana.setVisible(true);


        pmp.actualiza();
        LoggerDouble lgCV=LoggerFactory.nuevoLoggerDouble(controlador, "comandoVolante", 1000/250);
        LoggerArrayDoubles lgError=LoggerFactory.nuevoLoggerArrayDoubles(controlador, "error", 1000/250);
        lgError.setDescripcion("[error angular,error laterar]");
        Logger lgInstantes=LoggerFactory.nuevoLoggerTiempo(controlador, "Ciclo");
        LoggerArrayDoubles lgTrayectoria=LoggerFactory.nuevoLoggerArrayDoubles(controlador,"Posicion");
        lgTrayectoria.setDescripcion("[pos X, Pos y]");
        LoggerArrayDoubles lgDeseada=LoggerFactory.nuevoLoggerArrayDoubles(controlador, "Deseada");
        lgDeseada.setDescripcion("Punto más cercano [X,Y]");
        Logger lgParadas=null;
        
        //Una ves definidos todos, abrimos ventana de Loggers
        vl=new VentanaLoggers();
        boolean caminando=false;
        int indice = 0;
        while (true) {
            if (jcbCaminar.isSelected()) {
            	if(!caminando) { //acaba de activarse
            		//LoggerFactory.activaLoggers();
            		caminando=true;
            	}
            	lgInstantes.add();
                double comandoVolante = controlador.calculaComando();
                lgCV.add(comandoVolante);
                if (comandoVolante > COTA_ANGULO) {
                    comandoVolante = COTA_ANGULO;
                }
                if (comandoVolante < -COTA_ANGULO) {
                    comandoVolante = -COTA_ANGULO;
                //System.out.println("Comando " + comandoVolante);
                }
                carroOri.setConsignaVolante(comandoVolante);
                carroViejo.setConsignaVolante(comandoVolante);   
                carroOri.setPostura(carroOri.getX(),carroOri.getY(),carroOri.getTita(),carroOri.getVolante());
                lgTrayectoria.add(carroOri.getX(),carroOri.getY());
                carroOri.calculaEvolucion(comandoVolante, 2, 0.2);
                carroViejo.calculaEvolucion(comandoVolante, 2, 0.2);
                indice=controlador.indMinAnt;
//                System.out.println(indice);
                double errorAngular = rutaPruebaRellena.rumbo[indice] - carroOri.getTita();
                lgError.add(errorAngular, controlador.distanciaLateral);
                lgDeseada.add(rutaPruebaRellena.x[indice],rutaPruebaRellena.y[indice]);
                //System.out.println("Error " + error);
                pmp.actualiza();
            } else {
            	if(caminando) { //acaba de desactivarse
            		//LoggerFactory.vuelcaLoggersMATv4("Datos/PanelMuestra");
            		//LoggerFactory.vuelcaLoggersOctave("Datos/PanelMuestra");
            		//Para probar cuando se añade logger a posteriori
            		if(lgParadas==null)
            			lgParadas=LoggerFactory.nuevoLoggerTiempo(pmp, "Paradas");
            		lgParadas.add();
            		caminando=false;
            	}
            }
            try {
                Thread.sleep(250);
            } catch (Exception e) {
            }

        }

    }
}

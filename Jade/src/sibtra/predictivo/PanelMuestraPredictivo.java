package sibtra.predictivo;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Label;
import java.awt.event.MouseEvent;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;

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
	
	/** Puntero al controlador predicctivo a mostrar */
	ControlPredictivo CP=null;
	
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

	/** Constructor necesita el controlador predictivo */
	public PanelMuestraPredictivo(ControlPredictivo contPredic) {
		super();
		CP=contPredic;
		setTr(CP.ruta);
		{//nuevo panel para añadir debajo
			JPanel jpPre=new JPanel(new FlowLayout(FlowLayout.LEADING));
			//barra de progreso para el comando
			jpbComandoI=new JProgressBar(-pbMax,0);
			jpPre.add(jpbComandoI);

			jlComando=new JLabel("+##.##º");
			jlComando.setHorizontalAlignment(SwingConstants.TRAILING);
			//jlComando.setMinimumSize(new Dimension(50, 20));
			jlComando.setPreferredSize(new Dimension(60, 20));
			jpPre.add(jlComando);

			//barra de progreso para el comando derecho
			jpbComandoD = new JProgressBar(0,pbMax);
			jpPre.add(jpbComandoD);

			jlDistancia=new JLabel("DL: +###.##");
			//jlDistancia.setMinimumSize(new Dimension(60, 20));
			jlDistancia.setPreferredSize(new Dimension(80, 20));
			//jlDistancia.setSize(new Dimension(60, 20));
			jlDistancia.setHorizontalAlignment(SwingConstants.TRAILING);
			jpPre.add(jlDistancia);

			jpPre.add(new Label("H Pred"));
			jsModHPred=new SpinnerNumberModel(1,1,25,1);
			JSpinner jsHorPred=new JSpinner(jsModHPred);
			jsModHPred.addChangeListener(this);
			jpPre.add(jsHorPred);

			jpPre.add(new Label("H Cont"));
			jsModHCont=new SpinnerNumberModel(1,1,25,1);
			JSpinner jsHorCont=new JSpinner(jsModHCont);
			jsModHCont.addChangeListener(this);
			jpPre.add(jsHorCont);

			jpPre.add(new Label("Landa"));
			jsModLanda=new SpinnerNumberModel(0,0,100,0.1);
			JSpinner jsLanda=new JSpinner(jsModLanda);
			jsModLanda.addChangeListener(this);
			jpPre.add(jsLanda);
			
			jpPre.setMinimumSize(new Dimension(Short.MAX_VALUE,40));
//			jpPre.setBorder(BorderFactory.createCompoundBorder(
//	                   BorderFactory.createLineBorder(Color.red),
//	                   jpPre.getBorder()));

			add(jpPre);
		}
	}
	
	/** Lo que añadimos al panel */
	protected void cosasAPintar(Graphics g0) {
		//colocamos el coche en su posición actual
		situaCoche(CP.carroOriginal.getX(), CP.carroOriginal.getY()
		           , CP.carroOriginal.getTita());
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		
		//pintamos la trayectoria predicha
		GeneralPath gpPred=pathArrayXY(CP.prediccionPosicion);
		if(gpPred!=null) {
			g.setStroke(new BasicStroke());
			g.setColor(Color.WHITE);
			g.draw(gpPred);
		}
		
		//pintamos la orientación final
		{
			double[] pini=CP.prediccionPosicion[CP.horPrediccion-1];
			double ori=CP.predicOrientacion[CP.horPrediccion-1];
			g.setStroke(new BasicStroke());
			g.setColor(Color.RED);			
			double[] pfin={pini[0]+largoFlecha*Math.cos(ori)
					,pini[1]+largoFlecha*Math.sin(ori) };
			g.draw(new Line2D.Double(point2Pixel(pini),point2Pixel(pfin)));
	
		}
	}

	/** Atiende cambios en los spiners de los parámetros del controlador */
	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==jsModHPred) {
			CP.setHorPrediccion(jsModHPred.getNumber().intValue());
		}
		if(ce.getSource()==jsModHCont) {
			int hc=jsModHCont.getNumber().intValue();
			int hp=jsModHPred.getNumber().intValue();
			if(hc>hp)
				jsModHCont.setValue(hp);
			CP.setHorControl(hc);
		}
		if(ce.getSource()==jsModLanda) {
			CP.setLanda(jsModLanda.getNumber().doubleValue());
		}
	}
	
	/** programa el repintado del panel actulizando los valores de los spiners, etiquetas etc.*/
	public void actualiza() {
		//barra de progreso con el comando
		if(CP.comandoCalculado>0) {
			jpbComandoI.setValue(pbMax);
			jpbComandoD.setValue((int)(CP.comandoCalculado*pbMax/(Math.PI/4)));
		} else {
			jpbComandoD.setValue(0);
			jpbComandoI.setValue((int)(CP.comandoCalculado*pbMax/(Math.PI/4)));			
		}
		//texto con el comando
		jlComando.setText(String.format("%+04.2fº", Math.toDegrees(CP.comandoCalculado)));
		//texto con la distancia
		jlDistancia.setText(String.format("DL=%+04.2f", CP.distanciaLateral));
		
		//reflejamos valores usados por controlador
		jsModHCont.setValue(CP.horControl);
		jsModHPred.setValue(CP.horPrediccion);
		jsModLanda.setValue(CP.landa);
		
		super.actualiza();
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
        final Coche carroOri = new CocheModeloAntiguo();
        double vel = 2;
        double consVolante = 0;
        carroOri.setVelocidad(vel);
        carroOri.setConsignaVolante(consVolante);
        int horPredic = 12;
        int horCont = 3;
        double paramLanda = 1;
        double paramTs = 0.2;
//        double[][] rutaPrueba = ControlPredictivo.generaRuta(200,0.25);
        int numPuntos=200;
        double[][] rutaPrueba = new double[numPuntos][3];
        rutaPrueba[0][0] = 0;
        rutaPrueba[0][1] = 0;
        rutaPrueba[0][2] = 0;
        for (int i=1;i<numPuntos;i++){
            rutaPrueba[i][0] = 40*Math.cos(i*2*Math.PI/numPuntos)-40;
            rutaPrueba[i][1] = 40*Math.sin(i*2*Math.PI/numPuntos);
//            rutaPrueba[i][1] = rutaPrueba[i][0]*2;
//            rutaPrueba[i][1] = 3*Math.sin(rutaPrueba[i][0]*2*Math.PI/20);
            rutaPrueba[i][2] = Math.atan2((rutaPrueba[i][1]-rutaPrueba[i-1][1]),
                                        (rutaPrueba[i][0]-rutaPrueba[i-1][0]));
        }

        carroOri.setPostura(0,+10,0.5,0.0);
//        carroOri.setPostura(rutaPrueba[2][0],rutaPrueba[2][1],rutaPrueba[2][2]+0.3,0);
        
        ControlPredictivo controlador = new ControlPredictivo(carroOri,rutaPrueba,
                                            horPredic,horCont,paramLanda,paramTs);
        //ventana
		JFrame ventana=new JFrame("Panel Muestra Predictivo");		
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelMuestraPredictivo pmp=new PanelMuestraPredictivo(controlador){
			/** Evento cuando se pulsó el ratón con el SHIFT, establece la posición deseada */
			MouseEvent evenPos;

			/**
		     * Sólo nos interesan pulsaciones del boton 1. 
		     * Con CONTROL para determinar posición y orientación. Sin nada para hacer zoom.
		     * @see #mouseReleased(MouseEvent)
		     */
			public void mousePressed(MouseEvent even) {
				evenPos=null;
				if(even.getButton()==MouseEvent.BUTTON1 && (even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0) {
					//Punto del coche
					Point2D.Double nuevaPos=pixel2Point(even.getX(),even.getY());
					System.out.println("Pulsado Boton 1 con CONTROL "+even.getButton()
							+" en posición: ("+even.getX()+","+even.getY()+")"
							+"  ("+nuevaPos.getX()+","+nuevaPos.getY()+")  "
					);
					evenPos=even;
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
				if(even.getButton()==MouseEvent.BUTTON1 
						&& (even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0
							&& evenPos!=null) {
						System.out.println("Soltado con Control Boton "+even.getButton()
								+" en posición: ("+even.getX()+","+even.getY()+")");
						//Creamos rectángulo si está suficientemente lejos
						if(Math.abs(even.getX()-evenPos.getX())>50 
								|| Math.abs(even.getY()-evenPos.getY())>50) {
							Point2D.Double nuevaPos=pixel2Point(evenPos.getX(),evenPos.getY());
							Point2D.Double posAngulo=pixel2Point(even.getX(),even.getY());
							double yaw=Math.atan2(nuevaPos.getY()-posAngulo.getY(), nuevaPos.getX()-posAngulo.getX());
					        carroOri.setPostura(nuevaPos.getX(),nuevaPos.getY(),yaw,0.0);
					        actualiza();

						}
						return;
					}
				//al final llamamos al del padre
				super.mouseReleased(even);
			}


		};

		ventana.add(pmp);
		JCheckBox jcbCaminar=new JCheckBox("Caminar");
		jcbCaminar.setSelected(false);
		ventana.add(jcbCaminar,BorderLayout.PAGE_END);
		ventana.setSize(new Dimension(900,700));
		ventana.setVisible(true);
		



//		for (int i = 0; i < rutaPrueba.length; i++) {
		pmp.actualiza();
		while (true) {
			if(jcbCaminar.isSelected()) {
				double comandoVolante = controlador.calculaComando(); 
				if (comandoVolante > Math.PI/4)
					comandoVolante = Math.PI/4;
				if (comandoVolante < -Math.PI/4)
					comandoVolante = -Math.PI/4;
				//System.out.println("Comando " + comandoVolante);
				carroOri.setConsignaVolante(comandoVolante);
				carroOri.calculaEvolucion(comandoVolante,2,0.2);
				int indice = ControlPredictivo.calculaDistMin(rutaPrueba,carroOri.getX(),carroOri.getY());
				double error = rutaPrueba[indice][2] - carroOri.getTita();
				//System.out.println("Error " + error);
				pmp.actualiza();
			}
			try {
				Thread.sleep(250);
			} catch (Exception e) { }

        }

	}


}

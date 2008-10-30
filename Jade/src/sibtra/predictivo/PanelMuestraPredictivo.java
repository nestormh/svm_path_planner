package sibtra.predictivo;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Label;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;

import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
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
	public JCheckBox jcbCaminar;
	private SpinnerNumberModel jsModPred;
	private JSpinner jsHorPred;
	private SpinnerNumberModel jsModCont;
	private JSpinner jsHorCont;
	private SpinnerNumberModel jsModLanda;
	private JSpinner jsLanda;

	/** Constructor necesita el controlador predictivo */
	public PanelMuestraPredictivo(ControlPredictivo contPredic) {
		super();
		CP=contPredic;
		setTr(CP.ruta);
		//consas a añadir al jpSur
		//barra de progreso para el comando
		jpbComandoI=new JProgressBar(-pbMax,0);
		jpSur.add(jpbComandoI);
		
		jlComando=new JLabel("+##.##º");
		jlComando.setMinimumSize(new Dimension(80, 20));
		jlComando.setPreferredSize(new Dimension(80, 45));

		jpSur.add(jlComando);
		//barra de progreso para el comando derecho
		jpbComandoD = new JProgressBar(0,pbMax);
		jpSur.add(jpbComandoD);
		
		jlDistancia=new JLabel("+###.##");
		jlDistancia.setMinimumSize(new Dimension(80, 20));
		jlDistancia.setPreferredSize(new Dimension(80, 45));
		jpSur.add(jlDistancia);
		
		jcbCaminar=new JCheckBox("Caminar");
		jcbCaminar.setSelected(false);
		jcbCaminar.addActionListener(this);
		jpSur.add(jcbCaminar);

		//jpSur.add(new Label("H Pred"));
		jsModPred=new SpinnerNumberModel(10,1,25,1);
		jsHorPred=new JSpinner(jsModPred);
		jsHorPred.addChangeListener(this);
		jpSur.add(jsHorPred);
		
		jsModCont=new SpinnerNumberModel(3,1,25,1);
		jsHorCont=new JSpinner(jsModCont);
		jsHorCont.addChangeListener(this);
		jpSur.add(jsHorCont);

		jsModLanda=new SpinnerNumberModel(1,0,100,0.1);
		jsLanda=new JSpinner(jsModLanda);
		jsLanda.addChangeListener(this);
		jpSur.add(jsLanda);
}
	
	/** Lo que añadimos al panel */
	protected void cosasAPintar(Graphics g0) {
		//colocamos el coche en su posición actual
		situaCoche(CP.prediccionPosicion[0][0], CP.prediccionPosicion[0][1]
		           , CP.predicOrientacion[0]);
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
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
        Coche carroOri = new Coche();
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

        carroOri.setPostura(0,-10,0.5,0.0);
//        carroOri.setPostura(rutaPrueba[2][0],rutaPrueba[2][1],rutaPrueba[2][2]+0.3,0);
        
        ControlPredictivo controlador = new ControlPredictivo(carroOri,rutaPrueba,
                                            horPredic,horCont,paramLanda,paramTs);
        //ventana
		JFrame ventana=new JFrame("Panel Muestra Predictivo");		
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelMuestraPredictivo pmp=new PanelMuestraPredictivo(controlador);
		ventana.add(pmp);
		ventana.setSize(new Dimension(1024,800));
		ventana.setVisible(true);

//		for (int i = 0; i < rutaPrueba.length; i++) {
		pmp.actualiza();
		while (true) {
			if(pmp.jcbCaminar.isSelected()) {
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

	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==jsHorPred) {
			CP.setHorPrediccion(jsModPred.getNumber().intValue());
		}
		if(ce.getSource()==jsHorCont) {
			int hc=jsModCont.getNumber().intValue();
			int hp=jsModPred.getNumber().intValue();
			if(hc>hp)
				jsModCont.setValue(hp);
			CP.setHorControl(hc);
		}
		if(ce.getSource()==jsLanda) {
			CP.setLanda(jsModLanda.getNumber().doubleValue());
		}
	}

}

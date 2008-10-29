package sibtra.predictivo;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JProgressBar;

import sibtra.util.PanelMuestraTrayectoria;

/**
 * Panel para mostrar la información del control predictivo y poder depurarlo.
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelMuestraPredictivo extends PanelMuestraTrayectoria {

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
		jpSur.add(jlComando);
		//barra de progreso para el comando derecho
		jpbComandoD = new JProgressBar(0,pbMax);
		jpSur.add(jpbComandoD);
		
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
        double[][] rutaPrueba = ControlPredictivo.generaRuta(200,0.25);
        carroOri.setPostura(-1.0,0.0,0.5,0.0);
//        carroOri.setPostura(rutaPrueba[2][0],rutaPrueba[2][1],rutaPrueba[2][2]+0.3,0);
        
        ControlPredictivo controlador = new ControlPredictivo(carroOri,rutaPrueba,
                                            horPredic,horCont,paramLanda,paramTs);
        //ventana
		JFrame ventana=new JFrame("Panel Muestra Predictivo");		
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelMuestraPredictivo pmp=new PanelMuestraPredictivo(controlador);
		ventana.add(pmp);
		ventana.setSize(new Dimension(800,600));
		ventana.setVisible(true);

		for (int i = 0; i < rutaPrueba.length; i++) {            
            double comandoVolante = controlador.calculaComando(); 
            if (comandoVolante > Math.PI/4)
                comandoVolante = Math.PI/4;
            if (comandoVolante < -Math.PI/4)
                comandoVolante = -Math.PI/4;
            System.out.println("Comando " + comandoVolante);
            carroOri.setConsignaVolante(comandoVolante);
            carroOri.calculaEvolucion(comandoVolante,2,0.2);
            int indice = ControlPredictivo.calculaDistMin(rutaPrueba,carroOri.getX(),carroOri.getY());
            double error = rutaPrueba[indice][2] - carroOri.getTita();
            System.out.println("Error " + error);
			pmp.actualiza();
			try {
				Thread.sleep(500);
			} catch (Exception e) { }

        }

	}

}

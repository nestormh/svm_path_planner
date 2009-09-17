package sibtra.ui.modulos;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;

import javax.swing.AbstractAction;
import javax.swing.JButton;

import sibtra.lms.BarridoAngular;
import sibtra.lms.PanelMuestraBarrido;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
import sibtra.util.ThreadSupendible;

public class PerseguidorRF implements CalculoDireccion, CalculoVelocidad {
	String NOMBRE="Perseguidor";
	String DESCRIPCION="Calcula el angulo del volante y la velocidad para aproximarse al objetivo mas cercano";
	private VentanasMonitoriza ventanaMonitoriza;
	private BarridoAngular ultimoBarrido=null;
	private double[] angDistRF={0,80};
	private double consignaDir=0.0;
	private double distancia=Double.POSITIVE_INFINITY;
	private PanelPerseguidor panel;
	private int indMinAnt = -1;
	private ThreadSupendible thActulizacion;
	private int rangoInd = 10;
	private double velCrucero = 2;
	private int indInf=-1;
	private int indSup=-1;
	private boolean terminado=false;

	public double getConsignaDireccion() {		
		return consignaDir;
	}
	
	public double getConsignaVelocidad() {
		double velocidad = velCrucero;
		if (angDistRF[1] > 5){
			velocidad = velCrucero + angDistRF[1]*0.05;
		}		
		return velocidad;
	}
	
	/**
	 * Se encarga de encontrar el punto más cercano al coche (supuesto objetivo para 
	 * ser perseguido)
	 * @param ba barrido completo del rangeFinder
	 * @return distancia y ángulo en el que se encuentra el objetivo (obstáculo más cercano)
	 */
	public double[] getAnguloDistObjetivo(BarridoAngular ba){
		double distMin = Double.POSITIVE_INFINITY;
		double[] anguloDistRF = new double[2];
		int indMinDist = 0;
		indInf=0;
		indSup=ba.numDatos();
		if (indMinAnt > 0){  // Búsqueda en torno al ángulo donde se detectó el objetivo anteriormente
			indInf = ((indMinAnt - rangoInd ) < 0)?0:(indMinAnt - rangoInd);
			indSup = ((indMinAnt + rangoInd) > ba.numDatos())?ba.numDatos():(indMinAnt + rangoInd);
		}		
		for(int i=indInf; i < indSup;i++){
			if (ba.getDistancia(i)< distMin){
				indMinDist = i;
				distMin = ba.getDistancia(i);
			}
		}
		indMinAnt = indMinDist;
		anguloDistRF[0] = ba.getAngulo(indMinDist);
		anguloDistRF[1] = ba.getDistancia(indMinDist);
		return anguloDistRF;
	}
	

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}

	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null && ventMonitoriza!=ventanaMonitoriza)
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		if(ventMonitoriza==ventanaMonitoriza)
			//el la misma, no hacemos nada ya que implementa 2 interfaces y puede ser elegido 2 veces
			return true;
		ventanaMonitoriza=ventMonitoriza;
		panel=new PanelPerseguidor();
		ventanaMonitoriza.añadePanel(panel,NOMBRE,true,false);
		thActulizacion=new ThreadSupendible() {
			@Override
			protected void accion() {
				ultimoBarrido=ventanaMonitoriza.conexionRF.esperaNuevoBarrido(ultimoBarrido);
				angDistRF = getAnguloDistObjetivo(ultimoBarrido);
				consignaDir = angDistRF[0]-Math.PI/2;
				distancia = angDistRF[1];
				panel.actualiza();
			}
		};
		thActulizacion.setName(NOMBRE);
		thActulizacion.activar();
		

		return true;
	}

	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		if(terminado) return; //ya fue terminado
		thActulizacion.terminar();
		ventanaMonitoriza.quitaPanel(panel);
		terminado=true;
	}
	
	@SuppressWarnings("serial")
	protected class PanelPerseguidor extends PanelMuestraBarrido {
		private PanelFlow panelInformacion;
			
		public PanelPerseguidor()  {
			super((short)80);
			{//nuevo panel para añadir debajo
				panelInformacion=new PanelFlow();
				panelInformacion.añadeAPanel(new LabelDatoFormato(PerseguidorRF.class,"getConsignaDirGrados","%6.2f º"), "Ang RF");
				panelInformacion.añadeAPanel(new LabelDatoFormato(PerseguidorRF.class,"getDistancia","%6.2f m"), "Dist RF");
				panelInformacion.añadeAPanel(new SpinnerDouble(PerseguidorRF.this,"setVelCrucero",0.05,4,0.05), "Vel Crucero");
				panelInformacion.añadeAPanel(new SpinnerInt(PerseguidorRF.this,"setRangoInd",1,100,1), "Rang Ind");
				IniciaBusqueda iniBusqueda = new IniciaBusqueda();
				panelInformacion.añadeAPanel(new JButton(iniBusqueda), "Reiniciar Búsqueda");			
				add(panelInformacion);
			}
		}
		
		protected void cosasAPintar(Graphics g0) {
			super.cosasAPintar(g0);
			Graphics2D g=(Graphics2D)g0;
			if(ultimoBarrido==null || indInf<0 || indSup<0) return;
			g.setColor(Color.MAGENTA);
			//linea a donde comienza y termina la exploración y el seleccionado
			Point2D.Double pxCentro=point2Pixel(0.0,0.0);			
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ultimoBarrido.getPunto(indInf))));
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ultimoBarrido.getPunto(indSup-1))));
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ultimoBarrido.getPunto(indMinAnt))));
			//Marcamos en rojo la parte explorada del camino
			g.setStroke(new BasicStroke(3));
			g.setColor(Color.RED);
			GeneralPath perimetro = 
				new GeneralPath(GeneralPath.WIND_EVEN_ODD,indSup-indInf );
			Point2D.Double px=point2Pixel(ultimoBarrido.getPunto(indInf));
			perimetro.moveTo((float)px.getX(),(float)px.getY());
			for(int i=indInf+1; i<indSup; i++ ) {
				px=point2Pixel(ultimoBarrido.getPunto(i));
				perimetro.lineTo((float)px.getX(),(float)px.getY());
			}
			g.draw(perimetro);
		}
		
		/**
		 * Para cambiar el barrido que se está mostrando.
		 * y actualiza la presentación
		 */
		public void actualiza() {
			if(ultimoBarrido!=null) 		
				setBarrido(ultimoBarrido);
			panelInformacion.actualizaDatos(PerseguidorRF.this);
			super.actualiza();
		}
	}

	public double getConsignaDirGrados() {
		return Math.toDegrees(consignaDir);
	}

	public double getDistancia() {
		return distancia;
	}
	
	@SuppressWarnings("serial")
	class IniciaBusqueda extends AbstractAction{
		
		public IniciaBusqueda(){
			super("Reinicia la búsqueda");
			setEnabled(true);
		}

		public void actionPerformed(ActionEvent e) {
			indMinAnt = -1;
			setEnabled(true);			
		}
		
	}

	public double getVelCrucero() {
		return velCrucero;
	}

	public void setVelCrucero(double velCrucero) {
		this.velCrucero = velCrucero;
	}

	/**
	 * @return el rangoInd
	 */
	public int getRangoInd() {
		return rangoInd;
	}

	/**
	 * @param rangoInd el rangoInd a establecer
	 */
	public void setRangoInd(int rangoInd) {
		this.rangoInd = rangoInd;
	}
	
}

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
import sibtra.util.ThreadSupendible;

public class PerseguidorRF implements CalculoDireccion, CalculoVelocidad {
	String NOMBRE="Perseguidor RF";
	String DESCRIPCION="Calcula el angulo del volante para aproximarse al objetivo mas cercano";
	private VentanasMonitoriza ventanaMonitoriza;
	private BarridoAngular ba=null;
	private double[] angDistRF={0,80};
	private double consignaDir;
	private double distancia;
	private PanelPerseguidor panel;
	private int indMinAnt = -1;
	private ThreadSupendible thActulizacion;
	private int rangoInd = 10;
	private double velCrucero = 2.5;
	private int indInf;
	private int indSup;
	private boolean terminado=false;

	public double getConsignaDireccion() {		
		return consignaDir;
	}
	
	public double getConsignaVelocidad() {
		double velocidad = velCrucero;
		if (angDistRF[1] > 5){
			velocidad = velCrucero + angDistRF[1]*0.2;
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
		if(ventanaMonitoriza!=null && ventMonitoriza!=ventanaMonitoriza) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar en otra ventana");
		}
		if(ventMonitoriza==ventanaMonitoriza)
			//el la misma, no hacemos nada ya que implementa 2 interfaces y puede ser elegido 2 veces
			return true;
		ventanaMonitoriza=ventMonitoriza;
		panel=new PanelPerseguidor();
		ventanaMonitoriza.añadePanel(panel,NOMBRE,true,false);
		thActulizacion=new ThreadSupendible() {
			BarridoAngular ba=null;
			@Override
			protected void accion() {
				ba=ventanaMonitoriza.conexionRF.esperaNuevoBarrido(ba);
				angDistRF = getAnguloDistObjetivo(ba);
				consignaDir = angDistRF[0]-Math.PI/2;
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
				IniciaBusqueda iniBusqueda = new IniciaBusqueda();
				panelInformacion.añadeAPanel(new JButton(iniBusqueda), "Reiniciar Búsqueda");			
				add(panelInformacion);
			}
		}
		
		protected void cosasAPintar(Graphics g0) {
			super.cosasAPintar(g0);
			Graphics2D g=(Graphics2D)g0;
			if(ba==null) return;
			g.setColor(Color.MAGENTA);
			//linea a donde comienza y termina la exploración y el seleccionado
			Point2D.Double pxCentro=point2Pixel(0.0,0.0);			
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ba.getPunto(indInf))));
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ba.getPunto(indSup-1))));
			g.draw(new Line2D.Double(pxCentro,point2Pixel(ba.getPunto(indMinAnt))));
			//Marcamos en rojo la parte explorada del camino
			g.setStroke(new BasicStroke(3));
			g.setColor(Color.RED);
			GeneralPath perimetro = 
				new GeneralPath(GeneralPath.WIND_EVEN_ODD,indSup-indInf );
			Point2D.Double px=point2Pixel(ba.getPunto(indInf));
			perimetro.moveTo((float)px.getX(),(float)px.getY());
			for(int i=indInf+1; i<indSup; i++ ) {
				px=point2Pixel(ba.getPunto(i));
				perimetro.lineTo((float)px.getX(),(float)px.getY());
			}
			g.draw(perimetro);
		}
		
		/**
		 * Para cambiar el barrido que se está mostrando.
		 * y actualiza la presentación
		 */
		public void actualiza() {
			if(ba!=null) 		
				super.setBarrido(ba);
			panelInformacion.actualizaDatos(PerseguidorRF.this);
			actualiza();
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
	
}

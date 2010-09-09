package sibtra.lidar;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;

import javax.swing.Box;
import javax.swing.JCheckBox;

import sibtra.util.PanelBarrido;

@SuppressWarnings("serial")
public class PanelMuestraBarridoAngular extends PanelBarrido {

	
	/**
	 * Barrido actual
	 */
	protected BarridoAngular barridoAct = null;
	
	protected JCheckBox jcbBarrido;

	protected Lidar lidar=null;

	public PanelMuestraBarridoAngular(short distanciaMaxima) {
		super(distanciaMaxima);

		jpChecks.add(Box.createHorizontalStrut(15));

		jcbBarrido=new JCheckBox("Barrido",true);
		jcbBarrido.addChangeListener(this);
		jpChecks.add(jcbBarrido);
		jcbBarrido.setEnabled(false);
		

	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
	
		if(barridoAct!=null && barridoAct.numDatos()>0 && jcbBarrido.isSelected()) {
			//pasamos a representar el barrido
			//Color más oscuro si los datos no son actuales
			if(hayDatosNuevos)
				g.setColor(Color.GREEN);
			else
				g.setColor(Color.GREEN.darker());
			g.setStroke(new BasicStroke());
			GeneralPath perimetro = 
				new GeneralPath(GeneralPath.WIND_EVEN_ODD, barridoAct.numDatos());
	
			Point2D.Double px=point2Pixel(barridoAct.getPunto(0));
			perimetro.moveTo((float)px.getX(),(float)px.getY());
			for(int i=1; i<barridoAct.numDatos(); i++) {
				px=point2Pixel(barridoAct.getPunto(i));
				//Siguientes puntos son lineas
				perimetro.lineTo((float)px.getX(),(float)px.getY());
			}
			g.draw(perimetro);
		}
	
	}

	
	/**
	 * Para cambiar el barrido que se está mostrando y actuliza presentación.
	 * @param barr barrido nuevo
	 */
	public void setBarrido(BarridoAngular barr) {
		barridoAct=barr;
		
		if(barr==null) {
			jcbBarrido.setEnabled(false);
			return;
		}
		jcbBarrido.setEnabled(true);
		hayDatosNuevos=true;
		repinta();
	}

	/** Fija el valor del {@link Lidar} del que tomar los datos. Puede ser <code>null</code> */
	public void setLidar(Lidar ldr) {
		lidar=ldr;
	}

	/** Actualiza campos con datos del {@link #lidar} */
	public void actualizaPeriodico() {
		BarridoAngular barrNuevo=(lidar!=null) ? lidar.getBarridoAngular() : null;
		hayDatosNuevos=(barrNuevo!=null) && ( (barridoAct==null) ||  (barrNuevo.sysTime!=barridoAct.sysTime) ) ;
		barridoAct=barrNuevo;
		super.actualizaPeriodico();
	}

	/** Lo que hace en th de espera en cada iteración */
	protected void actualizaEspera() {
		if(lidar==null) {
			System.err.println(getClass().getName()+": Tratando de hacer actualizacion de espera cuando orienta = null");
			try { Thread.sleep(1000); } catch (Exception e) { }  //perdemos tiempo
		}
		barridoAct=lidar.esperaNuevoBarrido(barridoAct);
		setBarrido(barridoAct);
	}

}
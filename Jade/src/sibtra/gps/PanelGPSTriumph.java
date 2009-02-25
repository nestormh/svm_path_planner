package sibtra.gps;

import java.awt.FlowLayout;
import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;

/**
 * Panel para mostrar información y gestionar GPS Triumph. 
 * La información se muestra usando {@link PanelMuestraGPSData} y se añaden
 * widget específicos de Triumph.
 * @author alberto
 *
 */
public class PanelGPSTriumph extends JPanel {
	
	PanelMuestraGPSData pmGdata;
	
	GPSConnectionTriumph gpsCT;

	private int cuentaEnlaceUltimo;

	private PanelDatos jpEnlace;
	
	public PanelGPSTriumph(GPSConnectionTriumph gct) {
		if(gct==null)
			throw new IllegalArgumentException("Conexion a GPS no puede ser null");
		gpsCT=gct;
		setLayout(new BoxLayout(this,BoxLayout.PAGE_AXIS));
		
		pmGdata=new PanelMuestraGPSData();
		add(pmGdata);
		{ //linea con datos de calidad del enlace
			jpEnlace=new PanelDatos(new FlowLayout(FlowLayout.CENTER));
			LabelDato lda=new LabelDatoFormato("  ### %  ",GPSConnectionTriumph.class,"getCalidadLink","%4.0f %%");
//			lda.setPreferredSize(new Dimension(100,50));
			jpEnlace.añadeAPanel(lda, "Cali Enlace",jpEnlace);
			lda=new LabelDatoFormato("   ####   ",GPSConnectionTriumph.class,"getNumOKLink"," %10d");
//			lda.setPreferredSize(new Dimension(100,50));
			jpEnlace.añadeAPanel(lda, "Paq OK Enlace",jpEnlace);
			add(jpEnlace);
		}
	}

	public void actualizaGPS(GPSData ngpsdt) {
		pmGdata.actualizaPunto(ngpsdt);
		if(cuentaEnlaceUltimo!=gpsCT.getNumOKLink()) {
			jpEnlace.actualizaDatos(gpsCT);
			cuentaEnlaceUltimo=gpsCT.getNumOKLink();
		} else
			jpEnlace.actualizaDatos(null);
	}

	/**
	 * @see sibtra.gps.GpsEventListener#handleGpsEvent(sibtra.gps.GpsEvent)
	 */
	public void handleGpsEvent(GpsEvent ev) {
		actualizaGPS(ev.getNuevoPunto());
		repinta();
	}

	/** programamos la actualizacion del panel */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});
	}
	
}

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
public class PanelGPSTriumph extends PanelMuestraGPSData {
	
	GPSConnectionTriumph gpsCT;

	private int cuentaEnlaceUltimo;

	private LabelDatoFormato ldCal;

	private LabelDatoFormato ldOK;

	public PanelGPSTriumph(GPSConnectionTriumph gct) {
		super(true); //solo espaciales
		if(gct==null)
			throw new IllegalArgumentException("Conexion a GPS no puede ser null");
		gpsCT=gct;
		//linea con datos de calidad del enlace
		ldCal=new LabelDatoFormato("  ### %  ",GPSConnectionTriumph.class,"getCalidadLink","%4.0f %%");
//		lda.setPreferredSize(new Dimension(100,50));
		añadeAPanel(ldCal, "Cali Enlace");
		ldOK=new LabelDatoFormato("   ####   ",GPSConnectionTriumph.class,"getNumOKLink"," %10d");
//		lda.setPreferredSize(new Dimension(100,50));
		añadeAPanel(ldOK, "Paq OK Enlace");
	}

	public void actualizaGPS(GPSData ngpsdt) {
		actualizaPunto(ngpsdt);
		if(cuentaEnlaceUltimo!=gpsCT.getNumOKLink()) {
			//TODO pueden no ser necesarios
			ldCal.setEnabled(true);
			ldOK.setEnabled(true);
			cuentaEnlaceUltimo=gpsCT.getNumOKLink();
		} else {
			ldCal.setEnabled(false);
			ldOK.setEnabled(false);
		}
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

package sibtra.gps;

import javax.swing.SwingUtilities;

import sibtra.util.LabelDatoFormato;

/**
 * Panel para mostrar información y gestionar GPS Triumph. 
 * La información se muestra usando {@link PanelMuestraGPSData} y se añaden
 * widget específicos de Triumph.
 * Define {@link #actualiza()}, por lo que permite programar auto-refresco.
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

	/** Actualiza en base al ultimo punto temporal */
	protected void actualiza() {
		//TODO actualizacion ldCAL etc.
		GPSData ngpsdt=gpsCT.getPuntoActualTemporal();
		actualizaPunto(ngpsdt);
		if(cuentaEnlaceUltimo!=gpsCT.getNumOKLink()) {
			ldCal.Actualiza(gpsCT,true);
			ldOK.Actualiza(gpsCT,true);
			cuentaEnlaceUltimo=gpsCT.getNumOKLink();
		} else {
			ldCal.Actualiza(gpsCT,false);
			ldOK.Actualiza(gpsCT,false);
		}
		super.actualiza();
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

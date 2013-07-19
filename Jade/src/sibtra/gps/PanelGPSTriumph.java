package sibtra.gps;

import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JFileChooser;
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
@SuppressWarnings("serial")
public class PanelGPSTriumph extends PanelMuestraGPSData {
	
	GPSConnectionTriumph gpsCT;

	private int cuentaEnlaceUltimo;

	private LabelDatoFormato ldCal;

	private LabelDatoFormato ldOK;
	
	private AbstractAction accionSalvaPutoActual;
	
	private JFileChooser fc;

	public PanelGPSTriumph(GPSConnectionTriumph gct) {
		super(true); //solo espaciales
		if(gct==null)
			throw new IllegalArgumentException("Conexion a GPS no puede ser null");
		gpsCT=gct;
		//linea con datos de calidad del enlace
		ldCal=new LabelDatoFormato(GPSConnectionTriumph.class,"getCalidadLink","%4.0f %%");
//		lda.setPreferredSize(new Dimension(100,50));
		añadeAPanel(ldCal, "Cali Enlace");
		ldOK=new LabelDatoFormato(GPSConnectionTriumph.class,"getNumOKLink"," %10d");
//		lda.setPreferredSize(new Dimension(100,50));
		añadeAPanel(ldOK, "Paq OK Enlace");
		//Botón salvar posición
		accionSalvaPutoActual=new AccionSalvaPosicion();
		JButton jbSalvaPos=new JButton(accionSalvaPutoActual);
		añadeAPanel(jbSalvaPos,"Grabar posición");
		
        //elegir fichero
        fc = new JFileChooser(new File("./Sitios"));
		
	}

	/**
	 * @return the accionSalvaPutoActual
	 */
	public AbstractAction getAccionSalvaPutoActual() {
		return accionSalvaPutoActual;
	}

	class AccionSalvaPosicion extends AbstractAction {

		public AccionSalvaPosicion() {
			super("Grabar Posición");
		}

		public void actionPerformed(ActionEvent arg0) {
			int devuelto = fc.showSaveDialog(PanelGPSTriumph.this);
			if (devuelto == JFileChooser.APPROVE_OPTION) {
				try {
					File file = fc.getSelectedFile();
					GPSData ptoAct=gpsCT.getPuntoActualTemporal();
					//                gpsCon.saveRuta(file.getAbsolutePath());
					ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
					oos.writeObject(ptoAct);
					oos.close();
				} catch (IOException ioe) {
					System.err.println("Error al escribir en el fichero con posición actual");
					System.err.println(ioe.getMessage());
				}

			}
		}
		
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
//		System.out.print("G");
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

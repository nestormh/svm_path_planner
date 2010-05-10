/**
 * 
 */
package sibtra.ui.modulos;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.io.File;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;

import sibtra.gps.EditaFicherosRuta;
import sibtra.gps.GPSData;
import sibtra.gps.GestionFlota;
import sibtra.gps.PanelEligeDestino;
import sibtra.gps.Tramos;
import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.SeleccionTrayectoriaInicial;

/**
 * @author alberto
 *
 */
public class EligeDestino  implements SeleccionTrayectoriaInicial {

	private VentanasMonitoriza ventanaMonitorizar;
	private GestionFlota gf;
	private JFileChooser jfc;
	private PanelEligeDestino ped;
	private AccionCargarTramos act;
	private JDialog ventEligeDest;
	
	public String getDescripcion() {
		return "Crea trayectoria seleccionando el destino";
	}

	public String getNombre() {
		return "Elige Destino";
	}

	public EligeDestino() {
		super();
	}
	
	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza vmonitoriza) {
		ventanaMonitorizar=vmonitoriza;
		
		//Creamos la gestión de flota
		gf=new GestionFlota();
		jfc=new JFileChooser(new File("./Rutas/Trayectorias"));

		act=new AccionCargarTramos(); 
		act.actionPerformed(null);

		ped=new PanelEligeDestino(gf);
		ventEligeDest=new JDialog(ventanaMonitorizar.ventanaPrincipal,"Elige Destino",true);
		ventEligeDest.add(ped);
        //Fijamos su tamaño y posición
        ventEligeDest.setBounds(0, 384, 1024, 742);
		ventEligeDest.getContentPane().add(new JButton(new AbstractAction("Usar Destino Elegido") {
			
			public void actionPerformed(ActionEvent e) {
				ventEligeDest.setVisible(false);
			}
			
		}),BorderLayout.SOUTH);
		
		return true;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.SeleccionRuta#getTrayectoria()
	 */
	public Trayectoria getTrayectoria() {
        GPSData pa = ventanaMonitorizar.conexionGPS.getPuntoActualTemporal();
        double x,y,angAct;
        if(pa==null) {
        	System.err.println("Modulo "+getNombre()+":No tenemos punto GPS con que hacer los cáclulos");
        	//TODO Para probar
        	x=102.20870818546146; y=-35.55660199644231; angAct=0; 
//        	return null;
        } else {
        	//sacamos los datos del GPS
        	x=pa.getXLocal();
        	y=pa.getYLocal();
        	angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + ventanaMonitorizar.getDesviacionMagnetica();
        }
        ped.situaCoche(x, y, angAct);
		ped.habilitaEleccionDestino(true);
		ventEligeDest.setVisible(true);
		double[] de=ped.getDestinoElegido();
		if(de==null) {
			System.out.println("No se eligió destino :-(");
			return null;
		} 
		
		return gf.trayectoriaADestino(x, y, angAct, ped.getDestinoElegido()[0], ped.getDestinoElegido()[1]);
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		// TODO Auto-generated method stub

	}
	
	/** Acción para la cargar nuevo fichero de tramos en {@link GestionFlota} */
	class AccionCargarTramos extends AbstractAction {

		public AccionCargarTramos() {
			super("Abrir Fichero Tramos");
		}
		
		public void actionPerformed(ActionEvent ae) {
			jfc.setCurrentDirectory(new File("./Rutas/Tramos"));
			int devuelto=jfc.showOpenDialog(ventanaMonitorizar.ventanaPrincipal);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=jfc.getSelectedFile();
				gf.cargaTramos(file,ventanaMonitorizar.conexionGPS.posicionDeLaBase());
			}
		}
	}

}

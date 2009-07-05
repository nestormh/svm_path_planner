/**
 * 
 */
package sibtra.ui;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JOptionPane;

import sibtra.ui.modulos.SeleccionRuta;
import sibtra.util.ClasesEnPaquete;
import sibtra.util.PanelFlow;
import sibtra.util.PanelMuestraTrayectoria;

/**
 * @author alberto
 *
 */
public class PanelTrayectoria extends PanelMuestraTrayectoria {


	private VentanasMonitoriza ventanaMonitorizar;
	private PanelFlow panelInferior;
	private AccionCambiarRuta accCambiaRuta;
	private Class[] arrClasSelecRuta;
	private String[] arrNomClasMotor;
	private SeleccionRuta obSelRuta=null;
	private double[][] trayectoriaActual=null;
	private AccionCambiarModulo accCambiaModulo;

	public PanelTrayectoria(VentanasMonitoriza monitoriza) {
		super();
		ventanaMonitorizar=monitoriza;
		
		//buscamos los modulos SeleccionaRuta y sus nombres
		arrClasSelecRuta=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.SeleccionRuta", "sibtra.ui.modulos");
		arrNomClasMotor=ClasesEnPaquete.nombreClases(arrClasSelecRuta);

		
		panelInferior=new PanelFlow();
		accCambiaModulo=new AccionCambiarModulo();
		panelInferior.add(new JButton(accCambiaModulo));
		accCambiaRuta=new AccionCambiarRuta();
		panelInferior.add(new JButton(accCambiaRuta));
		
		
		add(panelInferior);
		ventanaMonitorizar.añadePanel(this, "Trayectoria", true,false);
		
		eligeModuloYTrayectoria();
		setTr(trayectoriaActual);
		actualiza();
	}
	
	class AccionCambiarModulo extends AbstractAction {

		public AccionCambiarModulo() {
			super("Cambiar Modulo");
		}
		
		public void actionPerformed(ActionEvent ae) {
			eligeModuloYTrayectoria();
			setTr(trayectoriaActual);
			actualiza();
		}
	}

	class AccionCambiarRuta extends AbstractAction {

		public AccionCambiarRuta() {
			super("Cambiar Ruta");
		}
		
		public void actionPerformed(ActionEvent ae) {
			trayectoriaActual=obSelRuta.getTrayectoria();
			setTr(trayectoriaActual);
			actualiza();
		}
	}

	protected void eligeModuloYTrayectoria() {
		if(arrNomClasMotor.length==0) {
			JOptionPane.showMessageDialog(ventanaMonitorizar.ventanaPrincipal,
			    "No hay modulos selectores de ruta",
			    "Sin modulos",
			    JOptionPane.ERROR_MESSAGE);
			return;
		}
		//presentamos ventana para elegir módulo que nos proporcinará la trayectoria
		int modSel=-1;
		do {
			modSel = JOptionPane.showOptionDialog(ventanaMonitorizar.ventanaPrincipal,
					"Pulsa el selecctor de ruta a utilizar",
					"Selector Ruta",
					JOptionPane.YES_NO_CANCEL_OPTION,
					JOptionPane.QUESTION_MESSAGE,
					(Icon)null,
					(Object[])arrNomClasMotor
					,(Object)arrNomClasMotor[0]);
			if(modSel==JOptionPane.CLOSED_OPTION && obSelRuta==null) {
				modSel = JOptionPane.showConfirmDialog(
						ventanaMonitorizar.ventanaPrincipal,
						"Se está solicitando Ruta. Si para elegir módulo. No para seguir sin ruta",
						"Que ruta",
						JOptionPane.YES_NO_OPTION);
			}
		} while(!(modSel>=0 && modSel<arrNomClasMotor.length) && modSel!=JOptionPane.NO_OPTION );
		if(!(modSel>=0 && modSel<arrNomClasMotor.length)) {
			//se opto por seguir sin cambiar nada
			return;
		}
		//instanciamos el modulo selecctor elegido
		try {
			obSelRuta=(SeleccionRuta)((Class<SeleccionRuta>)arrClasSelecRuta[modSel]).newInstance();
		} catch (InstantiationException e) {
			System.err.println(getClass().getName()+": No podemos instanciar objeto de la clase "+arrClasSelecRuta[modSel].getName());
			e.printStackTrace();
			obSelRuta=null;
			trayectoriaActual=null;
			return;
		} catch (IllegalAccessException e) {
			System.err.println(getClass().getName()+": No podemos acceder a la clase "+arrClasSelecRuta[modSel].getName());
			e.printStackTrace();
			obSelRuta=null;
			trayectoriaActual=null;
			return;
		}
		//tenemos el modulos, procedemos a inicializarlo
		obSelRuta.setVentanaMonitoriza(ventanaMonitorizar);
		trayectoriaActual=obSelRuta.getTrayectoria();

	}
	
	public double[][] getTrayectoria() {
		// TODO Auto-generated method stub
		return trayectoriaActual;
	}

}

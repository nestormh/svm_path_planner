/**
 * 
 */
package sibtra.ui;

import java.awt.event.ActionEvent;
import java.util.Vector;

import javax.swing.AbstractAction;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JOptionPane;

import sibtra.gps.PanelExaminaTrayectoria;
import sibtra.gps.Trayectoria;
import sibtra.ui.defs.SeleccionRuta;
import sibtra.ui.defs.UsuarioTrayectoria;
import sibtra.util.ClasesEnPaquete;
import sibtra.util.PanelFlow;

/**
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelTrayectoria extends PanelExaminaTrayectoria {

	/** Para apuntar todos los obetos que han pedido trayectoria y poder avisarlos si hay cambio */
	Vector<UsuarioTrayectoria> usanTr=new Vector<UsuarioTrayectoria>(10);  //espacio para unos 10 modulos
	private VentanasMonitoriza ventanaMonitorizar;
	private PanelFlow panelInferior;
	private AccionCambiarTrayectoria accCambiaTrayectoria;
	private Class[] arrClasSelecRuta;
	private String[] arrNomClasMotor;
	private SeleccionRuta obSelRuta=null;
	Trayectoria trayectoriaActual=null;
	private AccionCambiarModulo accCambiaModulo;

	public PanelTrayectoria(VentanasMonitoriza monitoriza) {
		super();
		ventanaMonitorizar=monitoriza;
		
		//buscamos los modulos SeleccionaRuta y sus nombres
		arrClasSelecRuta=ClasesEnPaquete.clasesImplementan("sibtra.ui.defs.SeleccionRuta", "sibtra.ui.modulos"
				,ventanaMonitorizar.panSelModulos.cargadorClases);
		arrNomClasMotor=ClasesEnPaquete.nombreClases(arrClasSelecRuta);

		
		panelInferior=new PanelFlow();
        ventanaMonitorizar.menuAcciones.addSeparator();
		//accion de cambiar de trayectoria
		accCambiaTrayectoria=new AccionCambiarTrayectoria();
		panelInferior.add(new JButton(accCambiaTrayectoria));
		ventanaMonitorizar.menuAcciones.add(accCambiaTrayectoria);
		//accion de cambiar de modulo
		accCambiaModulo=new AccionCambiarModulo();
		panelInferior.add(new JButton(accCambiaModulo));
		ventanaMonitorizar.menuAcciones.add(accCambiaModulo);
		
		
		add(panelInferior);
		ventanaMonitorizar.añadePanel(this, "Trayectoria", true,false);
		
		eligeModuloYTrayectoria();
		setTrayectoria(trayectoriaActual);
		actualiza();
	}
	
	class AccionCambiarModulo extends AbstractAction {

		public AccionCambiarModulo() {
			super("Cambiar Modulo");
		}
		
		public void actionPerformed(ActionEvent ae) {
			if(ventanaMonitorizar.panSelModulos.accionParar.isEnabled()) {
				//los modulos están activos => no podemos cambiar
				JOptionPane.showMessageDialog(ventanaMonitorizar.ventanaPrincipal,
					    "El motor esto activa no se puede cambiar modulo de trayectoria",
					    "Motor activo",
					    JOptionPane.ERROR_MESSAGE);
				return;
			}
				
			eligeModuloYTrayectoria();
			setTrayectoria(trayectoriaActual);
			actualiza();
		}
	}

	class AccionCambiarTrayectoria extends AbstractAction {

		public AccionCambiarTrayectoria() {
			super("Cambiar Trayectoria");
		}
		
		public void actionPerformed(ActionEvent ae) {
			if(ventanaMonitorizar.panSelModulos.accionParar.isEnabled()) {
				//los modulos están activos => no podemos cambiar
				JOptionPane.showMessageDialog(ventanaMonitorizar.ventanaPrincipal,
					    "El motor está activo no se puede cambiar trayectoria",
					    "Motor activo",
					    JOptionPane.ERROR_MESSAGE);
				return;
			}
			trayectoriaActual=obSelRuta.getTrayectoria();
			setTrayectoria(trayectoriaActual);
			//avisamos a todos los apuntados
			for(UsuarioTrayectoria uta: usanTr)
				uta.nuevaTrayectoria(trayectoriaActual);
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
	
	Trayectoria getTrayectoria(UsuarioTrayectoria objUsaTr) {
		//Lo apuntamos
		usanTr.add(objUsaTr);
		return trayectoriaActual;
	}

	void liberaTrayectoria(UsuarioTrayectoria objUsaTr) {
		if(!usanTr.remove(objUsaTr))
			System.err.println(getClass().getName()+": se intenta borrar objeto no apuntado: ("
					+objUsaTr.getClass().getName()+")="+objUsaTr);
	}

}

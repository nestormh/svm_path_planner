/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.util.ManejaJoystick;
import sibtra.util.PanelJoystick;
import sibtra.util.ThreadSupendible;

/**
 * Módulo que usa joystick para tomar velocidad y dirección
 * @author alberto
 *
 */
public class NavegaJoystick implements CalculoDireccion, CalculoVelocidad {
	
	private final static String NOMBRE="Navega Joystick";
	private final static String DESCRIPCION="Usa el Joystick para definir direccion y velocidad";
	protected static final long milisActulizacion = 200;
	
	private VentanasMonitoriza ventanaMonitoriza=null;
	private ManejaJoystick manJoy;
	private ThreadSupendible thCiclico;
	private PanelJoystick panJoy;
	/** Puede ser terminado varias veces, para no repetir el procedimiento */
	private boolean terminado=false;
	
	public NavegaJoystick() {};

	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMoni) {
		if(ventanaMonitoriza!=null && ventMoni!=ventanaMonitoriza) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar en otra ventana");
		}
		if(ventMoni==ventanaMonitoriza)
			//el la misma, no hacemos nada ya que implementa 2 interfaces y puede ser elegido 2 veces
			return true;
		ventanaMonitoriza=ventMoni;
		manJoy=new ManejaJoystick();
		panJoy=new PanelJoystick(manJoy);

		ventanaMonitoriza.añadePanel(panJoy, "Joystick", false,false);
		
		thCiclico=new ThreadSupendible() {
			@Override
			protected void accion() {
				panJoy.actualiza(); //esto ya realiza el pool
				try { Thread.sleep(milisActulizacion); } catch (InterruptedException e) {}
			}
		};
		thCiclico.setName(NOMBRE);
		thCiclico.activar();
		return true; //inicialización correcta
	}

	/** @return directamente la velocidad que me da {@link ManejaJoystick} */
	public double getConsignaVelocidad() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return manJoy.getVelocidad();
	}

	/** @return directamente el alfa que calcula {@link ManejaJoystick} */
	public double getConsignaDireccion() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return manJoy.getAlfa();
	}

	/**
	 * @see sibtra.ui.defs.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	/**
	 * @see sibtra.ui.defs.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/** Suspendemos el {@link #thCiclico} */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		if(terminado) return; //ya fue terminado
		thCiclico.terminar();
		ventanaMonitoriza.quitaPanel(panJoy);
		terminado=true;
	}

}

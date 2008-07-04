/**
 * 
 */
package sibtra.lms;

/**
 * @author alberto
 *
 */
public interface ManejaTelegramas {

	/**
	 * Para abrir y configurar puerto serie
	 * 
	 * @param NombrePuerto nombre del puerto serie a utilizar
	 */
	public abstract boolean ConectaPuerto(String NombrePuerto);

	/**
	 * Espera la llegada de un telegrama por la serial y extrae el mensaje
	 * @return mensaje contenido en telegrama
	 */
	public abstract byte[] LeeMensaje();

	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * Espera la confirmación de que ha sido bien recibido.
	 * 
	 * @return true si se recibe confirmación
	 */
	public abstract boolean EnviaMensaje(byte[] mensaje);
	
	/**
	 * Cierra el puerto
	 */
	public boolean cierraPuerto();


}